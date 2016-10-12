from __future__ import print_function

import sys
from menpofit.dpm import DPMLearner, DPMFitter
from menpofit.dpm.main import get_model
from resnet_feature_pyramid import ResnetFeaturePyramid
import numpy as np
import menpo.io as mio

def debugging():
    # im = mio.import_image('/vol/atlas/databases/aflw_ibug/face_52982.jpg', normalise=False)
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/multipie'
    resnet_fea_pyramid = ResnetFeaturePyramid()
    dpm_learner = DPMLearner(feature_pyramid=resnet_fea_pyramid)
    # normal_to_pie = dpm_learner.pie_tree_to_normal_tree()
    # pie_to_normal = []
    # for i in range(68):
    #     index = np.where(np.array(normal_to_pie)==i)[0][0]
    #     pie_to_normal.append(index)
    # for i in range(68):
    #     dpm_learner.train_part_fast('/vol/atlas/homes/ks3811/pickles/resnet/multipie', i)
    dpm_learner.train_component(pickle_dev, 0)
    # pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet'
    # model_name = 'parts_model_0.pkl'
    # model = get_model(pickle_dev, model_name, resnet_fea_pyramid)
    # image = mio.import_image('/vol/hci2/Databases/video/MultiPIE/session02/png/038/01/19_0/038_02_01_190_05.png', normalize=False)
    # response = DPMFitter.get_part_response(image, model, 0)

def train_parts():
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet'
    resnet_fea_pyramid = ResnetFeaturePyramid()
    dpm_learner = DPMLearner(feature_pyramid=resnet_fea_pyramid)
    for i in range(1, 68):
        dpm_learner.train_part('/vol/atlas/homes/ks3811/pickles/resnet', i)

def visualize_tree():
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/multipie'
    dpm_learner = DPMLearner()
    defs = dpm_learner.build_defs(pickle_dev, 0, 0)
    points = np.zeros((68, 2))
    tree = dpm_learner.get_aflw_tree(0)
    for depth in range(1, tree.maximum_depth + 1):
        for cv in tree.vertices_at_depth(depth):  # for each vertex in that level
            par = tree.parent(cv)
            # print(par)
            # print(cv)
            points[cv] = points[par, :] + defs[cv-1]
    # print(points)
    return points

def test_learn_model():
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet'
    model_name = 'parts_model_0.pkl'
    resnet_fea_pyramid = ResnetFeaturePyramid()
    model = get_model(pickle_dev, model_name, resnet_fea_pyramid)
    print(model)

if __name__ == "__main__":
    # sys.exit(visualize_tree())
    # sys.exit(train_parts())
    sys.exit(debugging())
    # sys.exit(test_learn_model())
