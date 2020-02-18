from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from .model import Model
from .utils import AttnLabelConverter

class Options:
  def __init__(self):
    self.character = '0123456789'
    self.Transformation = 'TPS'
    self.FeatureExtraction = 'ResNet'
    self.SequenceModeling = 'BiLSTM'
    self.Prediction = 'Attn'
    self.num_fiducial = 20
    self.input_channel = 1
    self.output_channel = 512
    self.hidden_size = 256
    self.imgH = 32
    self.imgW = 100
    self.PAD = False
    self.rgb = False
    self.batch_max_length = 2
    self.saved_model = '/content/best_accuracy.pth'
    self.image_folder = '/content/numbers/val'

class TextPredictor:

  def __init__(self): # (opt)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt = Options()

    converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    model_state_dict = torch.load(opt.saved_model)
    for old_key in list(model_state_dict.keys()):
      new_key = old_key[7:]
      model_state_dict[new_key] = model_state_dict.pop(old_key)

    with torch.no_grad():
      model = Model(opt)

      model.load_state_dict(model_state_dict) # , map_location=device
      model = model.to(device)
      model.eval()

  def predict(self, image, input_size):    

    with torch.no_grad():
      # if crop is tensor:
      image = transforms.ToPILImage()(image).convert("L")
      # else if opencv:
      # ... opencv -> PIL

      # todo: pozbyć się PIL

      image = image.resize(input_size, Image.BICUBIC)
      image = transforms.ToTensor()(image)
      image.sub_(0.5).div_(0.5) # normalization
      image = image.unsqueeze(0) # batch dimension

      batch_size = image.size(0)

      image = image.cuda()
      length_for_pred = torch.IntTensor([2] * batch_size).cuda()
      text_for_pred = torch.LongTensor(batch_size, 2 + 1).fill_(0).cuda()

      preds = model(image, text_for_pred, is_train=False)
      _, preds_index = preds.max(2)
      pred = converter.decode(preds_index, length_for_pred)[0]

      preds_prob = F.softmax(preds, dim=2)
      preds_max_prob, _ = preds_prob.max(dim=2)
      pred_max_prob = preds_max_prob[0]

      pred_EOS = pred.find('[s]')
      pred = pred[:pred_EOS]
      pred_max_prob = pred_max_prob[:pred_EOS]
      confidence_score = pred_max_prob.cumprod(dim=0)[-1].item()

    return pred, confidence_score