def get_config(model):
    if model == "small":
        return SmallConfig()
    if model == "small2":
        return SmallConfig2()
    if model == "small3":
        return SmallConfig3()
    elif model == "medium":
        return MediumConifg()
    elif model == "large":
        return LargeConfig()
    elif model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", model)


class SmallConfig(object):
    """Small config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5
    train_data_len = 42603942 # TODO

class SmallConfig2(object):
    """Small config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 2 # NOTE HERE
    num_steps = 20
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 0.7
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5

class SmallConfig3(object):
    """Small config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3 # NOTE HERE
    num_steps = 20
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5



class MediumConifg(object):
    """Medium config."""
    init_scale = 0.05 # scale to initialize LSTM weights
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 35
    hidden_size = 650
    max_epoch = 6
    max_max_epoch = 39
    keep_prob = 0.5
    lr_decay = 0.8
    batch_size = 20
    vocab_size = 10000


class LargeConfig(object):
    """Large config."""
    init_scale = 0.04 # scale to initialize LSTM weights
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 35
    hidden_size = 1500
    max_epoch = 14
    max_max_epoch = 55
    keep_prob = 0.35
    lr_decay = 1 / 1.15
    batch_size = 20
    vocab_size = 10000


class TestConfig(object):
    """Tiny config, for testing."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 1.0
    max_grad_norm = 1
    num_layers = 1
    num_steps = 2
    hidden_size = 2
    max_epoch = 1
    max_max_epoch = 1
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000

