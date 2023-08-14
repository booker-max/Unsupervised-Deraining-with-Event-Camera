import torch
import torchvision
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import shutil
import csv
import argparse
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
from pathlib import Path
import time
import sys
from data.dataset_n_n_e import SequenceDataset as SequenceDataset_train
from data.dataset_n_n_e_test import SequenceDataset as SequenceDataset_test
from torch.utils.data import DataLoader
import utils_rmfd
from tools import batch_PSNR, batch_SSIM, calculate_parameters
from networks.model import RMFD

###DDP
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


class trainer:

    def __init__(self, args, opts):

        self.args = args
        self.opts = opts

        self.train_txt_file = args["train_txt_file"]
        self.test_txt_file = args["test_txt_file"]
        self.h5_file = args["h5_file"]
        self.test_h5_file = args["test_h5_file"]
        self.model_dir = args['model_dir']
        self.log_dir = args['log_dir']
        self.record_dir = args["record_dir"]

        self.save_freq = args["save_freq"]
        self.save_threshold = args["save_threshold"]
        self.train_scene_num = args["train_scene_num"]
        self.test_scene_num = args["test_scene_num"]
        self.scene_types = args["scene_types"]
        self.sequence_length = args["sequence_length"]
        self.num_bins = args["num_bins"]

        self.root_dir = ""

    def start_record(self):
        
        print('*'*80)
        print("start recording in {}".format(self.record_dir))
        print('*'*80)
        
        headers_test = ["scene_id", 'psnr_avg', 'ssim_avg']

        self.test_csv_file = os.path.join(self.record_dir, 'test_record.csv')
        
        with open(self.test_csv_file, 'w', newline='') as f:
            record = csv.writer(f)
            record.writerow(headers_test)

    def save_checkpoint(self, filename):

        checkpoint = {
                'epoch': self.current_epoch,
                # 'state_dict': self.multi_model_res.state_dict(),
                'step_img': {x:self.log_im_step[x]+1 for x in self.log_im_step.keys()},
                }

        for name in self.model.train_model_names:

            if isinstance(name, str):
                net = getattr(self.model, name)

                checkpoint[name] = net.state_dict()

        for name in self.model.optimizer_names:

            if isinstance(name, str):
                optimizer = getattr(self.model, name)

                checkpoint[name] = optimizer.state_dict()
        
        torch.save(checkpoint, filename)

    def makedir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
        else:
            shutil.rmtree(directory)
            os.makedirs(directory)

    def open_tensorboard(self):
        self.writer = SummaryWriter(str(Path(self.log_dir)))
        
        if self.resume is None:
            self.log_im_step = {'train':0, 'test':0}
    
    def vis_events(self, voxel, mode):
        if mode == "train":
            voxel = voxel[0,:,:].detach().cpu().numpy()
        else:
            voxel = voxel[0,:,:].cpu().numpy()
        pos_events = np.maximum(voxel, 0)
        neg_events = np.minimum(voxel, 0)       
        
        image_rgb = np.stack(
                [
                    pos_events, 
                    np.zeros([pos_events.shape[0],pos_events.shape[1]], dtype="uint8"), 
                    -neg_events, 
                ], 0
            )
        return image_rgb

    def vis_events_1(self, voxel, mode):
        if mode == "train":
            voxel = voxel[-1,:,:].detach().cpu().numpy()
        else:
            voxel = voxel[-1,:,:].cpu().numpy()
        pos_events = np.maximum(voxel, 0)
        neg_events = np.minimum(voxel, 0)       
        
        image_rgb = np.stack(
                [
                    pos_events, 
                    np.zeros([pos_events.shape[0],pos_events.shape[1]], dtype="uint8"), 
                    -neg_events, 
                ], 0
            )
        return image_rgb

    def vis_events_sequence(self, voxel, mode="train"):

        trans = torchvision.transforms.ToTensor()

        image_rgb_sequence = []

        voxel = voxel.cpu().numpy()

        for i in range(voxel.shape[0]):
                
            voxel_bin = voxel[i,0,:,:]

            pos_events = np.maximum(voxel_bin, 0)
            neg_events = np.minimum(voxel_bin, 0)

            pos_idx = np.nonzero(pos_events)
            neg_idx= np.nonzero(neg_events)
            nozero_index = np.nonzero(voxel_bin)

            img_white = np.full((pos_events.shape[0], pos_events.shape[1], 3), fill_value=255, dtype="uint8")

            img_white[nozero_index[0], nozero_index[1], :] = 0
            img_white[pos_idx[0], pos_idx[1], 0] = 255
            img_white[neg_idx[0], neg_idx[1], -1] = 255

            img_white = trans(img_white)

            image_rgb_sequence.append(img_white)
        
        return image_rgb_sequence

    def train(self, local_rank, nprocs):

        print('*'*80)
        print('Begin training...')
        print('*'*80)

        best_psnr = 0
        best_ssim = 0

        self.local_rank = local_rank
        self.nprocs = nprocs

        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl")

        if dist.get_rank() == 0:
    
            self.resume = None
            self.makedir(self.model_dir)
            self.makedir(self.record_dir)
            self.start_record()
            self.open_tensorboard()

        self.model = RMFD(self.args, self.opts, self.local_rank)

        self.model.setup(self.opts)
        
        if dist.get_rank() == 0:
            
            all_num_params = 0

            print('\n=====================================================================')
            
            for name in self.model.train_model_names:

                if isinstance(name, str):
                    net = getattr(self.model, name)

                num_params = utils_rmfd.count_network_parameters(net)

                all_num_params += num_params

                print("===> Model {} has {} parameters".format(name, num_params))

            print("The whole Model has {} parameters".format(all_num_params))

            print('\n=====================================================================')
        
        train_dataset = SequenceDataset_train(self.h5_file, self.train_txt_file, self.sequence_length)
        train_sampler = DistributedSampler(train_dataset)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=(train_sampler is None), sampler=train_sampler, num_workers = 4, pin_memory = True)
        
        max_epoch = 2100
        best_count = 0

        for ii in range(self.opts.epoch_count, max_epoch):

            # start = time.time()

            self.model.train()

            self.current_epoch = ii

            if True:

                train_dataloader.sampler.set_epoch(self.current_epoch)

                if dist.get_rank() == 0:
                    Loss_epoch_consist, Loss_epoch_gen, Loss_epoch_dis, Loss_epoch_cont_layer, \
                    Loss_epoch_pos, Loss_epoch_neg, Loss_epoch_neg_MM, Loss_epoch_neg_E, Loss_epoch_neg_F, \
                    Loss_epoch_sum = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                
                start = time.time()

                for iteration, sequence in enumerate(train_dataloader):

                    rainy_frame = sequence["rainy"].to(self.local_rank)
                    rainy_event = sequence["rainy_events"].to(self.local_rank)
                    gan_event = sequence["gan_events"].to(self.local_rank)
                    frame_g2 = sequence["gt"][:,3:6].to(self.local_rank)
                    frame_gan_g2 = sequence["gan_gt"][:,3:6].to(self.local_rank)

                    input_data = {"Rain_frame": rainy_frame, "Rain_event": rainy_event, "clean": frame_g2, "gan_clean": frame_gan_g2, "gan_event": gan_event}

                    self.model.set_input(input_data)
                    self.model.optimize_parameters()

                    loss = self.model.get_losses()

                    if dist.get_rank() == 0:

                        Loss_epoch_consist += loss["consist"]
                        Loss_epoch_gen += loss["adv_gen"]
                        Loss_epoch_dis += loss["adv_dis"]
                        Loss_epoch_cont_layer += loss["contrstive_layer"]
                        Loss_epoch_pos += loss["pos"]
                        Loss_epoch_neg += loss["neg"]
                        Loss_epoch_neg_MM += loss["neg_MM"]
                        Loss_epoch_neg_E += loss["neg_E"]
                        Loss_epoch_neg_F += loss["neg_F"]
                        Loss_epoch_sum += loss["G_sum"]

                if dist.get_rank() == 0:

                    Loss_epoch_consist /= (iteration+1)
                    Loss_epoch_gen /= (iteration+1)
                    Loss_epoch_dis /= (iteration+1)
                    Loss_epoch_cont_layer /= (iteration+1)
                    Loss_epoch_pos /= (iteration+1)
                    Loss_epoch_neg /= (iteration+1)
                    Loss_epoch_neg_MM /= (iteration+1)
                    Loss_epoch_neg_E /= (iteration+1)
                    Loss_epoch_neg_F /= (iteration+1)
                    Loss_epoch_sum /= (iteration+1)

                    vis_event = rainy_event[0,:,:,:].unsqueeze(1)
                    event_sequence = vutils.make_grid(self.vis_events_sequence(vis_event), normalize=True, scale_each=True)
                    vis_gan_event = gan_event[0,:,:,:].unsqueeze(1)
                    gan_event_sequence = vutils.make_grid(self.vis_events_sequence(vis_gan_event), normalize=True, scale_each=True)

                    self.writer.add_image("Train GT Image", frame_g2[0, :, :, :], self.log_im_step['train'])
                    self.writer.add_image("Train GAN_GT Image", frame_gan_g2[0, :, :, :], self.log_im_step['train'])
                    self.writer.add_image("Train Rainy Image", rainy_frame[0, 3:6, :, :], self.log_im_step['train'])
                    self.writer.add_image("Train Predict Bg", self.model.Pred_bg[0, :, :, :], self.log_im_step['train'])
                    self.writer.add_image("Train Predict Rain", self.model.Pred_rl[0, :, :, :], self.log_im_step['train'])
                    self.writer.add_image("Train Rainy Events", event_sequence, self.log_im_step['train'])
                    self.writer.add_image("Train GAN Clean Events", gan_event_sequence, self.log_im_step['train'])

                    self.writer.add_scalar('Loss_Consistency', Loss_epoch_consist, ii)
                    self.writer.add_scalar('Loss_adv_gen', Loss_epoch_gen, ii)
                    self.writer.add_scalar('Loss_adv_dis', Loss_epoch_dis, ii)
                    self.writer.add_scalar('Loss_Contrastive_Layer', Loss_epoch_cont_layer, ii)
                    self.writer.add_scalars('Loss_Pos_Neg', {"Loss_Pos": Loss_epoch_pos, "Loss_Neg": Loss_epoch_neg}, ii)
                    self.writer.add_scalars('Loss_Combine', {"Neg_MM": Loss_epoch_neg_MM, "Neg_F": Loss_epoch_neg_F, "Neg_E": Loss_epoch_neg_E}, ii)
                    self.writer.add_scalar('Loss_G_sum', Loss_epoch_sum, ii)
                    self.writer.add_scalar('Learning_Rate', self.model.optimizers[0].param_groups[0]['lr'], ii)

                    self.log_im_step['train'] += 1

                    print('_'*100, flush = True)
                    log_str = 'Train: Epoch:{:02d}/{:02d}, Loss_G_sum:{:.2e}'
                    print(log_str.format(ii, (self.opts.n_epochs + self.opts.n_epochs_decay + 1), Loss_epoch_sum), flush=True)
                    print('_'*100, flush = True)
                
                self.model.update_learning_rate()

                print("Epoch time is {:f}".format(time.time() - start))

            ####validation
            if True:

                if self.current_epoch >= self.save_threshold and self.current_epoch % self.save_freq == 0:
                    
                    scene_types = self.args["scene_types"]
                    self.model.eval()
                    
                    # start_infer = time.time()
                    with torch.no_grad():

                        psnr_list, ssim_list = [], []

                        for current_type in scene_types:

                            # start = time.time()

                            val_dataset = SequenceDataset_test(self.test_h5_file, self.sequence_length, current_type)
                            val_data_loader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=False, num_workers = 4, pin_memory = True)

                            psnr_scene_list, ssim_scene_list = [], []

                            for i, sequence in enumerate(val_data_loader):

                                rainy_frame = sequence["rainy"].to(self.local_rank)
                                frame_g2 = sequence["gt"][:,3:6].to(self.local_rank)
                                rainy_event = sequence["rainy_events"].to(self.local_rank)

                                input_data = {"Rain_frame": rainy_frame, "Rain_event": rainy_event, "clean": frame_g2}

                                self.model.set_input_test(input_data)
                                self.model.forward()

                                psnr = batch_PSNR(self.model.Pred_bg, frame_g2, ycbcr=False)
                                ssim = batch_SSIM(self.model.Pred_bg, frame_g2, ycbcr=False)
                                
                                psnr_scene_list.append(psnr)
                                ssim_scene_list.append(ssim)

                            psnr_scene = np.mean(psnr_scene_list)
                            ssim_scene = np.mean(ssim_scene_list)

                            psnr_list.append(psnr_scene)
                            ssim_list.append(ssim_scene)

                            if dist.get_rank() == 0:

                                log_str = 'Test: Epoch:{:02d}/{:02d}, Currenet_type: {}, PSNR={:4.2f}, SSIM={:6.4f}'
                                print(log_str.format(ii, (self.opts.n_epochs + self.opts.n_epochs_decay+1), current_type, psnr_scene, ssim_scene), flush=True)
                                
                                # print("Inference time is {:f}".format(time.time() - start))

                                vis_event = rainy_event[0,:,:,:].unsqueeze(1)
                                event_sequence = vutils.make_grid(self.vis_events_sequence(vis_event), normalize=True, scale_each=True)

                                self.writer.add_image("Test GT Image", frame_g2[0, :, :, :], self.log_im_step['test'])
                                self.writer.add_image("Test Rainy Image", rainy_frame[:,3:6][0, :, :, :], self.log_im_step['test'])
                                self.writer.add_image("Test Predict Bg", self.model.Pred_bg[0, :, :, :], self.log_im_step['test'])
                                self.writer.add_image("Test Predict Rain", self.model.Pred_rl[0, :, :, :], self.log_im_step['test'])
                                        
                                self.writer.add_image("Test Rainy Events", event_sequence, self.log_im_step['test'])
                                
                                self.log_im_step['test'] += 1

                        psnr_avg = np.mean(psnr_list)
                        ssim_avg = np.mean(ssim_list)
                        
                        if dist.get_rank() == 0:

                            self.writer.add_scalar('psnr', psnr_avg, ii)
                            self.writer.add_scalar('ssim', ssim_avg, ii)

                            print('=====================================================================')
                            log_str = 'Test: Epoch:{:02d}/{:02d}, PSNR_avg={:4.2f}, SSIM_avg={:6.4f}'
                            print(log_str.format(ii, (self.opts.n_epochs + self.opts.n_epochs_decay+1), psnr_avg, ssim_avg), flush=True)
                            print('=====================================================================')

                        if psnr_avg > best_psnr:
                            
                            if dist.get_rank() == 0:
                                checkpoint = os.path.join(self.model_dir, "best.pth.tar")
                                self.save_checkpoint(checkpoint)

                            best_psnr = psnr_avg
                            best_ssim = ssim_avg

                        elif psnr_avg <= best_psnr and self.current_epoch > (self.opts.n_epochs + self.opts.n_epochs_decay):

                                best_count+=1
                        
                        if dist.get_rank() == 0:
                            print('=====================================================================')
                            print('Best PSNR: {:.2f}, Best SSIM: {:.4f}'.format(best_psnr, best_ssim))
                            print('=====================================================================')
                        
                        # print("The whole Inference time is {:f}".format(time.time() - start_infer))

                # print("Inference time is {:f}".format(time.time() - start))
            
            # print(f"Best count is {best_count}")

            if best_count >= 20:
                break

        if dist.get_rank() == 0:
            with open(self.test_csv_file, 'w', newline='') as f:
                record = csv.writer(f)
                record.writerow(["Best", format(best_psnr, '.2f'), format(best_ssim, '.4f')])

            self.writer.close()
