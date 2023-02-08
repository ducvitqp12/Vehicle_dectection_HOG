from glob import glob
from os.path import join
from types import SimpleNamespace as SNS

pklpath = SNS(
    svc='backup/svc-all3-hog12-luv.pkl',
    scaler='backup/scaler-all3-hog12-luv.pkl',

)
imgspath = './data'
cars_imgspath = glob(join(imgspath, 'vehicles/', '*.png'))
notcars_imgspath = glob(join(imgspath, 'non-vehicles/', '*.png'))

default = SNS(

    color_space='LUV',
    orient = 12,
    pix_per_cell = 8,
    cell_per_block = 2,
    hog_channel = 'ALL',
    train_size = (64, 64),
    spatial_size = (32, 32),
    hist_bins = 32,
    hog_feat = True, 
    hist_feat = True,
    spatial_feat = True,
)
defaults = {
    'color_space':default.color_space,
    'orient':default.orient,
    'pix_per_cell':default.pix_per_cell,
    'cell_per_block':default.cell_per_block,
    'hog_channel':default.hog_channel,
    'spatial_size':default.spatial_size,
    'hist_bins':default.hist_bins,
    'spatial_feat':default.spatial_feat,
    'hist_feat':default.hist_feat,
    'hog_feat':default.hog_feat,
}
