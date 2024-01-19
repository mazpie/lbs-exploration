import torch.nn as nn

def get_random_cnn(state_size, use_bn=False, use_ln=False):
    random_conv_net = get_conv_net(state_size, use_bn=use_bn, use_ln=use_ln)
    
    for param in random_conv_net.parameters():
        param.requires_grad = False
    
    return random_conv_net

def get_random_mlp(obs_size, state_size, use_bn=False, use_ln=False):
    if use_bn:
        random_net = nn.Sequential(
                        nn.Linear(obs_size, state_size, bias=False),
                        nn.BatchNorm1d(state_size)
                    )
    else:
        random_net = nn.Sequential(
                        nn.Linear(obs_size, state_size),
                    )

    if use_ln:
        random_net = nn.Sequential(
            random_net, 
            nn.LayerNorm(state_size)
        )


    for param in random_net.parameters():
        param.requires_grad = False

    return random_net

def get_conv_net(state_size, use_bn=False, use_ln=False, mnist_convnet=False):
    if mnist_convnet:
        return get_mnist_convnet()
    feature_output = 7 * 7 * 64
    if use_bn:
        conv_net = nn.Sequential(
                        nn.Conv2d(
                            in_channels=4,
                            out_channels=32,
                            kernel_size=8,
                            stride=4,
                            bias=False),
                        nn.BatchNorm2d(32),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2,
                            bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            bias=False),
                        nn.BatchNorm2d(64),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        nn.Linear(feature_output, state_size, bias=False),
                        nn.BatchNorm1d(state_size) 
                    )   
    else:
        conv_net = nn.Sequential(
                        nn.Conv2d(
                            in_channels=4,
                            out_channels=32,
                            kernel_size=8,
                            stride=4),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=4,
                            stride=2),
                        nn.LeakyReLU(),
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=64,
                            kernel_size=3,
                            stride=1),
                        nn.LeakyReLU(),
                        nn.Flatten(),
                        nn.Linear(feature_output, state_size),
                    )
    if use_ln:
        conv_net = nn.Sequential(
            conv_net, 
            nn.LayerNorm(state_size)
        )
    
    return conv_net

def get_mnist_convnet():
    return nn.Sequential(
                        nn.Conv2d(1, 32, 3, 1),
                        nn.Conv2d(32, 64, 3, 1),
                        nn.MaxPool2d(2),
                        nn.Flatten(),
                        nn.Linear(9216, 128)
            )
 