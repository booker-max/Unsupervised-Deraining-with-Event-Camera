from BaseBlock import *
import numpy as np

class EventEncoder(nn.Module):
    def __init__(self):
        super(EventEncoder, self).__init__()
        self.encode = nn.Sequential(
            Conv2D(in_ch=1, out_ch=16, kernel_size=3),
            Dense_block(in_ch=16, k=4),
            Conv2D(in_ch=32, out_ch=32, kernel_size=3),
            Dense_block(in_ch=32, k=8),
            ResidualBlock(channel_num=64),
            ResidualBlock(channel_num=64)      
        )
        self.clstm = ConvLSTM(input_dim=64, hidden_dim=64, kernel_size=(3, 3), num_layers=1)
    
    def forward(self, event):
        embeding = torch.stack([self.encode(event[:, i:i+1, :, :]) for i in range(event.shape[1])], dim=1)
        last_output, bi_last_output = self.clstm(embeding)
        idx = event.shape[1]//2
        nest = torch.cat([last_output[0][:,idx-1,:,:,:],bi_last_output[0][:,idx-1,:,:,:]],dim=1)
        return nest

def count_network_parameters(model):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])

    return N

if __name__ == "__main__":

    model = EventEncoder().cuda()
    input = torch.randn(2,6,32,32).cuda()

    out = model(input)

    print(count_network_parameters(model))



