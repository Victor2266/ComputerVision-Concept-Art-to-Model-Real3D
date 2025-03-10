import torch
import torch.nn.functional as F
import numpy as np
import time
import logging
import random
import os
import dataclasses
import torch.distributed as dist
import itertools
from torch.cuda.amp import autocast
from utils import exp_utils, loss_utils, vis_utils, train_utils, dist_utils
import itertools
from scripts.validator import transform_clip
import clip

logger = logging.getLogger(__name__)
EPOCH_LENGTH = 300 #10000


def train_epoch_wild_with_consistency(config, loader, loader_sv, model, optimizer, scheduler, scaler,
                epoch, output_dir, device, rank, perceptual_loss, wandb_run):
    time_meters = exp_utils.AverageMeters()
    loss_meters = exp_utils.AverageMeters()
    loss_meters_sv = exp_utils.AverageMeters()
    loss_meters_sv_consistency = exp_utils.AverageMeters()

    model.train()
    perceptual_loss.eval()
    clip_model, _ = clip.load("ViT-B/16", device=device)
    clip_model.eval()

    batch_end = time.time()
    iter_num = EPOCH_LENGTH * epoch
    #train_utils.set_iteration(loader_sv, iter_num)

    loader_sv_iter = iter(loader_sv)
    for batch_idx, sample in enumerate(loader):
        torch.cuda.empty_cache()
        
        if batch_idx % 10 == 0:  # Print memory stats
            print(f"Start of iteration {batch_idx}")
            print(f"Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            print(f"Memory cached: {torch.cuda.memory_reserved()/1e9:.2f}GB")

        # Ensure tensors are contiguous and in right dtype
        input_image = sample['input_image'].contiguous()
        rays_o = sample['rays_o'].contiguous()
        rays_d = sample['rays_d'].contiguous()
        
        # Clear cache before forward pass
        torch.cuda.empty_cache()

        # Add dimension checks
        print("Input shapes:")
        print(f"input_image: {sample['input_image'].shape}")
        print(f"rays_o: {sample['rays_o'].shape}")
        print(f"rays_d: {sample['rays_d'].shape}")

        try:
            sample_sv = next(loader_sv_iter)
        except StopIteration:
            #train_utils.set_iteration(loader_sv, iter_num)
            loader_sv_iter = iter(loader_sv)
            sample_sv = next(loader_sv_iter)

        if batch_idx > EPOCH_LENGTH:
            break
        iter_num = batch_idx + EPOCH_LENGTH * epoch

        sample = exp_utils.dict_to_cuda(sample, device)
        sample = loss_utils.get_loss_target(config, sample)
        sample_sv = exp_utils.dict_to_cuda(sample_sv, device)
        if config.dataset.sv_curriculum != 'none':
            sample_sv = train_utils.set_pose_curriculum(config, iter_num, sample_sv)
        sample_sv, sample_sv_consistency = loss_utils.get_loss_target_consistency(config, sample_sv)
        time_meters.add_loss_value('Data time', time.time() - batch_end)
        end = time.time()

        # --------------------------  multiview data training --------------------------  
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            print(f"Memory before model forward: {torch.cuda.memory_allocated()/1e9:.2f}GB")
            torch.cuda.synchronize()
            
            # Process views one at a time
            results_list = []
            for view_idx in range(sample['rays_o'].shape[1]):
                # Process smaller chunks of rays for each view
                h, w = sample['rays_o'].shape[2:4]
                chunk_size = 32 * 32  # Process 32x32 patches at a time
                
                view_results_rgb = []
                view_results_weight = []
                
                for start_idx in range(0, h*w, chunk_size):
                    end_idx = min(start_idx + chunk_size, h*w)
                    
                    # Reshape rays for chunked processing
                    rays_o_chunk = sample['rays_o'][:, view_idx].reshape(-1, 3)[start_idx:end_idx]
                    rays_d_chunk = sample['rays_d'][:, view_idx].reshape(-1, 3)[start_idx:end_idx]
                    
                    chunk_results = model(
                        sample['input_image'],
                        rays_o_chunk.unsqueeze(1),
                        rays_d_chunk.unsqueeze(1)
                    )
                    
                    view_results_rgb.append(chunk_results['images_rgb'].detach())
                    view_results_weight.append(chunk_results['images_weight'].detach())
                    
                    # Clear chunk results
                    del chunk_results
                    torch.cuda.empty_cache()
                
                # Combine chunks for this view
                view_rgb = torch.cat(view_results_rgb, dim=0).reshape(1, 1, h, w, -1)
                view_weight = torch.cat(view_results_weight, dim=0).reshape(1, 1, h, w, -1)
                
                results_list.append({
                    'images_rgb': view_rgb,
                    'images_weight': view_weight
                })
                
                # Clear view results
                del view_results_rgb, view_results_weight, view_rgb, view_weight
                torch.cuda.empty_cache()

            # Combine results from all views
            results = {
                'images_rgb': torch.cat([r['images_rgb'] for r in results_list], dim=1),
                'images_weight': torch.cat([r['images_weight'] for r in results_list], dim=1)
            }
            
            # Compute losses
            losses = loss_utils.get_losses(config, results, sample, perceptual_loss)
            total_loss = 0.0
            for k, v in losses.items():
                if 'loss' in k:
                    total_loss += losses[k.replace('loss_', 'weight_')] * v
                    loss_meters.add_loss_value(k, v.detach().item())
            total_loss = total_loss / config.train.accumulation_step

            # Clear results list
            del results_list
            torch.cuda.empty_cache()

        # Backward pass
        if config.train.use_amp:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # Timing and logging
        time_meters.add_loss_value('Loss time', time.time() - end)
        time_meters.add_loss_value('Batch time', time.time() - batch_end)
        end = time.time()

        # Print progress
        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time'].avg,
                loss_time=time_meters.average_meters['Loss time'].avg,
                batch_time_avg=time_meters.average_meters['Batch time'].avg
            )
            for k, v in loss_meters.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)

        # Visualizations
        if iter_num % config.vis_freq == 0 and rank == 0:
            # Make sure to detach tensors for visualization
            vis_utils.vis_seq(
                vid_clips=sample['images_target'].detach(),
                vid_masks=sample['masks_target'].detach(),
                recon_clips=results['images_rgb'].detach(),
                recon_masks=results['images_weight'].detach(),
                iter_num=iter_num,
                output_dir=output_dir,
                subfolder='train_seq',
                inv_normalize=False
            )
            torch.cuda.empty_cache()  # Clear after visualization

        # Finally, clean up
        del results, losses, total_loss
        torch.cuda.empty_cache()
        # --------------------------  multiview data training ends--------------------------  
        
        # --------------------------  single-view data training --------------------------  
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            results_sv = model(sample_sv['input_image'],
                            sample_sv['rays_o'],
                            sample_sv['rays_d'])
            time_meters.add_loss_value('Prediction time (SV)', time.time() - end)
            end = time.time()

            losses_sv = loss_utils.get_losses_sv(config, results_sv, sample_sv, perceptual_loss, clip_model)
            total_loss_sv = 0.0
            for k, v in losses_sv.items():
                if 'loss' in k:
                    total_loss_sv += losses_sv[k.replace('loss_', 'weight_')] * v
                    loss_meters_sv.add_loss_value(k, v.detach().item())
            total_loss_sv = total_loss_sv / config.train.accumulation_step

        # if config.train.use_amp:
        #     scaler.scale(total_loss_sv).backward(retain_graph=True)
        # else:
        #     total_loss_sv.backward(retain_graph=True)
        if config.train.use_amp:
            scaler.scale(total_loss_sv).backward()
        else:
            total_loss_sv.backward()


        time_meters.add_loss_value('Loss time (SV)', time.time() - end)
        time_meters.add_loss_value('Batch time (SV)', time.time() - batch_end)
        end = time.time()
        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time (SV)'].avg,
                loss_time=time_meters.average_meters['Loss time (SV)'].avg,
                batch_time_avg=time_meters.average_meters['Batch time (SV)'].avg
            )
            for k, v in loss_meters_sv.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)
        # --------------------------  single-view data training ends ---------------------------------

        # --------------------------  single-view self-consistency training --------------------------
        if config.train.rerender_consistency_input:
            with torch.no_grad():
                with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
                    results_sv_hres = model(sample_sv['input_image'],
                                            sample_sv['rays_o_hres'][:,-1:],
                                            sample_sv['rays_d_hres'][:,-1:])
        else:
            results_sv_hres = None
        sample_sv_consistency = loss_utils.format_consistency_input_output(config, 
                                                                           sample_sv_consistency, 
                                                                           sample_sv, 
                                                                           results_sv, results_sv_hres)
        with autocast(enabled=config.train.use_amp, dtype=torch.bfloat16):
            results_sv_cy = model(sample_sv_consistency['input_image'],
                                  sample_sv_consistency['rays_o'],
                                  sample_sv_consistency['rays_d'])
            time_meters.add_loss_value('Prediction time (CY)', time.time() - end)
            end = time.time()

            losses_sv_cy = loss_utils.get_losses_consistency(config, results_sv_cy, sample_sv_consistency, perceptual_loss, clip_model)
            total_loss_sv_cy = 0.0
            for k, v in losses_sv_cy.items():
                if 'loss' in k:
                    total_loss_sv_cy += losses_sv_cy[k.replace('loss_', 'weight_')] * v
                    loss_meters_sv_consistency.add_loss_value(k, v.detach().item())
            total_loss_sv_cy = total_loss_sv_cy / config.train.accumulation_step

        if config.train.use_amp:
            scaler.scale(total_loss_sv_cy).backward()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                scaler.step(optimizer)
                optimizer.zero_grad()
                scaler.update()
                scheduler.step()
        else:
            total_loss_sv_cy.backward()
            if (batch_idx+1) % config.train.accumulation_step == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.train.grad_max, norm_type=2.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        time_meters.add_loss_value('Loss time (CY)', time.time() - end)
        time_meters.add_loss_value('Batch time (CY)', time.time() - batch_end)
        if iter_num % config.print_freq == 0:  
            msg = 'Epoch {0}, Iter {1}, rank {2}, ' \
                'Time: data {data_time:.3f}s, pred {recon_time:.3f}s, loss {loss_time:.3f}s ({batch_time_avg:.3f}s), Loss: '.format(
                epoch, iter_num, rank,
                data_time=time_meters.average_meters['Data time'].avg,
                recon_time=time_meters.average_meters['Prediction time (CY)'].avg,
                loss_time=time_meters.average_meters['Loss time (CY)'].avg,
                batch_time_avg=time_meters.average_meters['Batch time (CY)'].avg
            )
            for k, v in loss_meters_sv_consistency.average_meters.items():
                tmp = '{0}: {loss.val:.4f} ({loss.avg:.4f}), '.format(
                        k, loss=v)
                msg += tmp
            msg = msg[:-2]
            logger.info(msg)
        # --------------------------  single-view self-consistency training ends--------------------------  

        if iter_num % config.vis_freq == 0 and rank == 0:
            n_sv, n_cy = results_sv['images_rgb'].shape[1], results_sv_cy['images_rgb'].shape[1]
            n = n_cy + n_sv
            vis_utils.vis_seq(vid_clips=sample_sv['images_target'].repeat(1,n,1,1,1),
                            vid_masks=sample_sv['masks_target'].repeat(1,n,1,1,1),
                            recon_clips=torch.cat([results_sv['images_rgb'], results_sv_cy['images_rgb']], dim=1),
                            recon_masks=torch.cat([results_sv['images_weight'], results_sv_cy['images_weight']], dim=1),
                            iter_num=iter_num,
                            output_dir=output_dir,
                            subfolder='train_seq_sv',
                            inv_normalize=False)
            
        if rank == 0 and wandb_run is not None:
            wandb_log = {'Train/loss': total_loss.item(),
                         'Train/loss_sv': total_loss_sv.item(),
                         'Train/loss_sv_cy': total_loss_sv_cy.item(),
                         'Train/lr': optimizer.param_groups[0]['lr']}
            for k, v in losses.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            for k, v in losses_sv.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            for k, v in losses_sv_cy.items():
                if 'loss' in k:
                    wandb_log['Train/{}'.format(k)] = v.item()
            wandb_run.log(wandb_log)


        if iter_num % 1000 == 0 and iter_num != 0 and batch_idx != 0:
            if config.train.use_zeroRO:
                print('Consolidated on rank {} because of ZeRO'.format(rank))
                optimizer.consolidate_state_dict(0)
            dist_utils.save_on_master({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict() if dist_utils.is_main_process() else None,
                'scaler': scaler.state_dict(),
                'schedular': scheduler.state_dict()
            }, save_path=os.path.join(output_dir, "cpt_last.pth.tar"))
        
        dist.barrier()
        batch_end = time.time()
    
    del losses, total_loss, results, sample

