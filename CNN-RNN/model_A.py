import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.optim as optim
import json

class EncoderCNN(nn.Module):
    def __init__(self, embed_size = 1024,train_CNN=False):
      super(EncoderCNN, self).__init__()
      self.train_CNN=False
      # get the pretrained densenet model
      self.densenet = models.densenet121(pretrained=True)
      # replace the classifier with a fully connected embedding layer
      self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
      # add another fully connected layer
      self.embed = nn.Linear(in_features=1024, out_features=embed_size)
      # dropout layer
      self.dropout = nn.Dropout(p=0.5)
      # activation layers
      self.prelu = nn.PReLU()

    def forward(self, images):
        # get the embeddings from the densenet
        densenet_outputs = self.dropout(self.prelu(self.densenet(images)))

        # pass through the fully connected
        embeddings = self.embed(densenet_outputs)
        return embeddings

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.lstm_cell = nn.LSTMCell(embed_size, hidden_size,num_layers)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
        self.dropout=nn.Dropout(0.5)
        # activations
        self.softmax = nn.Softmax(dim=1)

    def forward(self, features, captions):
        batch_size = features.size(0)
        # init the hidden and cell states to zeros
        hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        # define the output tensor placeholder
        outputs = torch.empty((batch_size, captions.shape[1], self.vocab_size)).cuda()

        # embed the captions
        captions_embed = self.dropout(self.embed(captions)).cuda()
        # print("Captions_embed size :- ",captions_embed.shape)
        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))

            # for the 2nd+ time step, using teacher forcer
            else:
                # print("hidden_state size :- ",captions_embed[:, t, :].shape)
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))

            # output of the attention mechanism
            out = self.fc_out(hidden_state).cuda()

            # build the output tensor
            outputs[:, t, :] = out.cuda()


        return outputs

    
class ImageCaptioner(nn.Module):
  def __init__(self, vocab_path,saved_model_path):
      super(ImageCaptioner, self).__init__()
      self.encoder = EncoderCNN()
      self.feature_size=1024
      with open(vocab_path, 'r') as json_file:
        self.vocabulary = json.load(json_file)
      self.decoder = DecoderRNN(self.feature_size,self.feature_size,len(self.vocabulary.items()))
      self.decoder.load_state_dict(torch.load(saved_model_path))
      self.processor=transforms.Compose([
          transforms.Resize((224, 224)),  # Resize image to match model input size
          transforms.ToTensor(),           # Convert image to tensor
          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image
        ])

  def predict(self,img_path):
    input_image = Image.open(img_path)
    input_image = input_image.convert('RGB')
    input_tensor = self.processor(input_image)
    input_batch = input_tensor.unsqueeze(0)
    with torch.no_grad():
      features = self.encoder.forward(input_batch)
    # vocabulary={i:word for i,word in enumerate(vocabulary)}
    states=None
    hiddens=None
    # features.shape
    max_words=20
    embed_hidst=features.clone()
    output_tokens=[]
    with torch.no_grad():
      for _ in range(max_words):
        if hiddens==None:
          hiddens, states = self.decoder.lstm_cell(features, states)
        else:
          hiddens, states = self.decoder.lstm_cell(embed_hidst, (hiddens,states))
        output = self.decoder.fc_out(hiddens.unsqueeze(0))
        max_index = torch.argmax(output)
        # print(max_index)
        output_tokens.append(max_index)
        max_index=max_index.unsqueeze (0)
        embed_hidst=self.decoder.embed(max_index)
    # print(output_tokens)
    text_words=[self.vocabulary[str(token.item())] for token in output_tokens]
    text_sentence = ' '.join(text_words)
    return text_sentence
