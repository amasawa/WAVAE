from sklearn import preprocessing
import sklearn

def TestSegment(preprocessing,x_raw,x_label,vqrae_output):
    #min_max_scaler = sklearn.preprocessing.MinMaxScaler()
    '''
    x_raw = [batch, seq, latent]
    x_raw[:,-1] = [batch, latent]
    '''
    min_max_scaler = sklearn.preprocessing.StandardScaler()
    if preprocessing:
        x_recon = min_max_scaler.fit_transform(vqrae_output.dec_means.detach().cpu().numpy()[:, -1])
        #x_recon = vqrae_output.dec_means.detach().cpu().numpy()[:, -1]
        x_raw = min_max_scaler.fit_transform(x_raw[:, -1])
        #x_raw = x_raw[:, -1]
        x_label = x_label[:, -1]
    else:
        x_recon = vqrae_output.dec_means.detach().cpu().numpy()[:, -1]
        x_raw = x_raw[:, -1]
        x_label = x_label[:, -1]
    return x_raw, x_label, x_recon


