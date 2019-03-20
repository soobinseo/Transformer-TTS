from preprocess import get_post_dataset, DataLoader, collate_fn_postnet
from network import *
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import os
from tqdm import tqdm

def adjust_learning_rate(optimizer, step_num, warmup_step=4000):
    lr = hp.lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
def main():

    dataset = get_post_dataset()
    global_step = 0
    
    m = nn.DataParallel(ModelPostNet().cuda())

    m.train()
    optimizer = t.optim.Adam(m.parameters(), lr=hp.lr)

    writer = SummaryWriter()

    for epoch in range(hp.epochs):

        dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_fn_postnet, drop_last=True, num_workers=8)
        pbar = tqdm(dataloader)
        for i, data in enumerate(pbar):
            pbar.set_description("Processing at epoch %d"%epoch)
            global_step += 1
            if global_step < 400000:
                adjust_learning_rate(optimizer, global_step)
                
            mel, mag = data
        
            mel = mel.cuda()
            mag = mag.cuda()
            
            mag_pred = m.forward(mel)

            loss = nn.L1Loss()(mag_pred, mag)
            
            writer.add_scalars('training_loss',{
                    'loss':loss,

                }, global_step)
                    
            optimizer.zero_grad()
            # Calculate gradients
            loss.backward()
            
            nn.utils.clip_grad_norm_(m.parameters(), 1.)
            
            # Update weights
            optimizer.step()

            if global_step % hp.save_step == 0:
                t.save({'model':m.state_dict(),
                                 'optimizer':optimizer.state_dict()},
                                os.path.join(hp.checkpoint_path,'checkpoint_postnet_%d.pth.tar' % global_step))

            
            


if __name__ == '__main__':
    main()