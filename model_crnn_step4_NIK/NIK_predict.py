import torch

from model_crnn_step4_NIK import models, params, dataset, utils

from torch.autograd import Variable

weight = '/home/huyvd/Documents/works/pytorch_tutorial/tap_code/indo_card_code/demo_main/weight/crnn_NIK.pth'
input_size = (params.imgW, params.imgH)

nclass = len(params.alphabet) + 1


class NIK_PredictCRNN:
    def __init__(self):
        self.model = models.CRNN(params.imgH, params.nc, nclass, params.nh)
        self.model.load_state_dict(torch.load(weight, map_location=torch.device('cpu')))
        self.model.eval()

    def predict(self, PILimg):

        img_transformed = dataset.resizeNormalize((190, 32))(PILimg)
        img_transformed = img_transformed.view(1, *img_transformed.size())
        input_img = Variable(img_transformed)
        preds = self.model(input_img)

        converter = utils.strLabelConverter(params.alphabet)
        preds_size = Variable(torch.LongTensor([preds.size(0)]))

        _, preds = preds.max(2)

        preds = preds.transpose(1, 0).contiguous().view(-1)  # dao chieu
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        return sim_pred
