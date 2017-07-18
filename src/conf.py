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
    elif model == "small6":
        return Small6Config()
    elif model == "small7":
        return Small7Config()
    elif model == "small8":
        return Small8Config()
    elif model == "hid1":
        return Hidden1Config()
    elif model == "hid2":
        return Hidden2Config()
    elif model == "hid3":
        return Hidden3Config()
    elif model == "hid4":
        return Hidden4Config()
    elif model == "hid5":
        return Hidden5Config()
    elif model == "hid6":
        return Hidden6Config()
    elif model == "hid7":
        return Hidden7Config()
    elif model == "hid8":
        return Hidden8Config()
    elif model == "hid9":
        return Hidden9Config()
    elif model == "proj1":
        return Proj1Config()
    elif model == "proj2":
        return Proj2Config()
    elif model == "medium3":
        return Medium3Conifg()
    elif model == "large":
        return LargeConfig()
    elif model == "test":
        return TestConfig()
    else:
        raise ValueError("Invalid model: %s", model)


class Proj1Config(object):
    """Projection config. Compared with Hidden7Config"""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 256 
    hidden_size = 1024 
    num_proj = 256 # NOTE HERE
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Proj2Config(object):
    """Projection config. Compared with Proj1Config"""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 256 
    hidden_size = 1024 
    num_proj = 512 # NOTE HERE
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5


class Hidden1Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 150
    hidden_size = 100 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden2Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 150
    hidden_size = 128 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden3Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 150
    hidden_size = 256 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden4Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 150
    hidden_size = 512 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden5Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 150
    hidden_size = 1024 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden6Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 256 # NOTE HERE
    hidden_size = 512 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden7Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 256 # NOTE HERE
    hidden_size = 1024 # NOTE HERE
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden8Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 3
    num_steps = 20
    embedding_size = 256 # NOTE HERE
    hidden_size = 512 # NOTE HERE
    num_proj = 256
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5

class Hidden9Config(object):
    """Hidden config."""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 2 # NOTE HERE
    num_steps = 20
    embedding_size = 256 # NOTE HERE
    hidden_size = 1024 # NOTE HERE
    num_proj = 256
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 
    vocab_size = 100000 + 2
    punc_size = 5


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

class Small6Config(object):
    """Small config. Compare with SmallConfig()"""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 32 # NOTE HERE
    vocab_size = 100000 + 2
    punc_size = 5

class Small7Config(object):
    """Small config. Compare with SmallConfig()"""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 64 # NOTE HERE
    vocab_size = 100000 + 2
    punc_size = 5

class Small8Config(object):
    """Small config. Compare with SmallConfig()"""
    init_scale = 0.1 # scale to initialize LSTM weights
    learning_rate = 0.1
    max_grad_norm = 5
    num_layers = 1
    num_steps = 20
    embedding_size = 100
    hidden_size = 100
    num_proj = 100
    max_epoch = 4
    max_max_epoch = 7
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 128 # NOTE HERE
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

