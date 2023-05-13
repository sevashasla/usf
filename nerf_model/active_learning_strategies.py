'''
strategies for active learning
'''
import torch
from abc import abstractclassmethod

class StrategyBase:
    def __init__(self, opt, model, holdout_dataset):
        self.opt = opt
        self.model = model
        self.holdout_dataset = holdout_dataset

    @abstractclassmethod
    def calculate_score(self):
        raise NotImplementedError()

    def __get_preds(self, data):
        rays_o = data['rays_o'] # [B, N, 3]
        rays_d = data['rays_d'] # [B, N, 3]
        images = data['images'] # [B, H, W, 3/4]

        assert images.size(0) == 1 # batch_size must be equal to 1

        # eval with fixed background color
        bg_color = 1

        outputs = self.model.render(rays_o, rays_d, staged=True, bg_color=bg_color, perturb=False, **vars(self.opt))

        pred_alpha = outputs['alphas']
        alphas_shifted = torch.cat([torch.ones_like(pred_alpha[..., :1]), 1 - pred_alpha + 1e-15], dim=-1) # [N, T+t+1]
        weights = pred_alpha * torch.cumprod(alphas_shifted, dim=-1)[..., :-1]

        return {
            "weights": weights,
            **outputs,
        }

    def choose_new_k(self, k):
        with torch.no_grad():
            scores = []
            su_weight = self.opt.su_weight

            self.model.eval()
            for data in self.holdout_dataset.dataloader():

                outputs = self.__get_preds(data)
                score = 0.0
                if self.opt.use_uncert:
                    score += (1 - su_weight) * self.calculate_score(
                        outputs["uncertainty_image"], 
                        outputs["uncertainty_all"], 
                        outputs["weights"]
                    ).item()
                    
                if self.opt.use_semantic_uncert:
                    score += su_weight * self.calculate_score(
                        outputs["semantic_uncertainty_image"], 
                        outputs["semantic_uncertainty_all"], 
                        outputs["weights"]
                    ).item()
                scores.append(score)
            
            scores = torch.tensor(scores)
            index = torch.topk(scores, k)[1].cpu().numpy()
            return index


class BayesianStrategy(StrategyBase):
    '''
    from ActiveNeRF
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculate_score(self, pred_uncert, uncert_all, weights):
        pre = uncert_all.sum([1,2])
        post = (1. / (1. / uncert_all + weights ** 2.0 / pred_uncert)).sum([1 , 2])
        return pre - post


class MeanStrategy(StrategyBase):
    '''
    much more simpler strategy
    bigger mean uncertainty -> must choose
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def calculate_score(self, pred_uncert, uncert_all, weights):
        return pred_uncert.mean()

### SIMPLE FUNCTIONS ###

def bayesian_choose(opt, model, holdout_dataset, k):
    strategy = BayesianStrategy(opt, model, holdout_dataset)
    return strategy.choose_new_k(k)

def mean_choose(opt, model, holdout_dataset, k):
    strategy = MeanStrategy(opt, model, holdout_dataset)
    return strategy.choose_new_k(k)
