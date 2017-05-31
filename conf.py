def get_config(model):
    if model == "small":
        return SmallConfig()
    elif model == "small2":
        return Small2Config()
    elif model == "small3":
        return Small3Config()
    elif model == "small4":
        return Small4Config()
    elif model == "small5":
        return Small5Config()
    elif model == "medium3":
        return Medium3Conifg()
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
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5

class Small2Config(object):
    """Small config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 2 # NOTE HERE
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5

class Small3Config(object):
    """Small config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3 # NOTE HERE
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5

class Small4Config(object):
    """Small config. Compared with SmallConfig."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 50 # NOTE HERE
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5

class Small5Config(object):
    """Small config. Compared with SmallConfig."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 150 # NOTE HERE
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5


class Medium3Conifg(object):
    """Medium config."""
    init_scale = 0.05 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 10
    keep_prob = 0.5
    lr_decay = 0.5
    batch_size = 256
    vocab_size = 100000 + 2
    punc_size = 5


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

