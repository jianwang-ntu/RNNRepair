
__all__=["create_classifer"]
import os 
from os.path import join as path_join 

def create_classifer(model_type,*args,**kwargs):
    assert "save_dir" in kwargs 
    assert "epoch" in kwargs 
    here_dir = os.path.dirname(__file__)
    dataset_default_dir=path_join(here_dir,"../../data/")
    
    classifier = None 
    if model_type == 'keras_lstm_mnist':
        from .image_classification.mnist_rnn_profile import MnistClassifier
        classifier = MnistClassifier(rnn_type='lstm', *args,**kwargs,)
    elif model_type == 'torch_lstm_bin':
        from .image_classification.mnist_rnn_binary import TorchMnistiClassifier
        classifier = TorchMnistiClassifier(rnn_type='lstm', *args,**kwargs,
                                           flip=0, first=4, second=9, ratio=0.3)

    
    elif model_type == 'torch_gru_imdb':
        dataset_path=kwargs.get("dataset_path",path_join(dataset_default_dir,"imdb_data"))
        from .sentiment_analysis.imdb_rnn_profile import IMDBClassifier
        classifier = IMDBClassifier(rnn_type='gru',dataset_path=dataset_path, *args,**kwargs,)

    elif model_type == 'torch_gru_toxic':
        dataset_path=kwargs.get("dataset_path",path_join(dataset_default_dir,"toxic_data"))
        from .sentiment_analysis.toxic_rnn_profile import TOXICClassifier
        classifier = TOXICClassifier(rnn_type='gru',dataset_path=dataset_path, *args,**kwargs,)

    elif model_type == 'torch_gru_sst':
        dataset_path=kwargs.get("dataset_path",path_join(dataset_default_dir,"sst_data"))
        from .sentiment_analysis.sst_rnn_profile import SSTClassifier
        classifier = SSTClassifier(rnn_type='gru',dataset_path=dataset_path, *args,**kwargs,)
    
    assert classifier is not None, f"expect the model_type in included into \
        [keras_lstm_mnist,torch_gru_imdb,torch_gru_toxic,torch_gru_sst,torch_lstm_bin], \
        your input is {model_type}" 

    return classifier
