class VRAEOutput(object):
    def __init__(self,  training_time=None, testing_time=None, zs=None, z_infer_means=None, z_infer_stds=None, decs=None, dec_means=None,
                 dec_stds=None, kld_loss=None, nll_loss=None):


        self.dec_means = dec_means
        self.dec_stds = dec_stds
        self.kld_loss = kld_loss
        self.nll_loss = nll_loss
        self.best_TN = None
        self.best_FP = None
        self.best_FN = None
        self.best_TP = None
        self.best_precision = None
        self.best_recall = None
        self.best_fbeta = None
        self.best_pr_auc = None
        self.best_roc_auc = None
        self.best_cks = None
        self.roc_list = []
        self.pr_list = []
        self.F1_list = []

        self.training_time = training_time
        self.testing_time = testing_time

