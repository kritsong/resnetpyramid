from __future__ import print_function

import sys
from menpofit.dpm import DPMLearner, DPMFitter
from menpofit.dpm.main import get_model
from resnet_feature_pyramid import ResnetFeaturePyramid
import numpy as np
import menpo.io as mio
from menpo.image import Image
import matplotlib.pyplot as plt

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
    for i in range(68):
        dpm_learner.train_part_fast('/vol/atlas/homes/ks3811/pickles/resnet/5x5_dpm_filters', i)
    # dpm_learner.train_component(pickle_dev, 0)
    # dpm_learner.train_final_component(pickle_dev, 0)
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
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/5x5_dpm_filters/pose_4'
    dpm_learner = DPMLearner()
    defs = dpm_learner.build_defs(pickle_dev, 0, 1)
    points = np.zeros((68, 2))
    tree = dpm_learner.get_aflw_tree(1)
    for depth in range(1, tree.maximum_depth + 1):
        for cv in tree.vertices_at_depth(depth):  # for each vertex in that level
            par = tree.parent(cv)
            # print(par)
            # print(cv)
            points[cv] = points[par, :] + defs[cv-1]
    # print(points)
    return points

def test_learn_model():
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/multipie'
    model_name = '10pos_component_0.pkl'
    resnet = ResnetFeaturePyramid()
    model = get_model(pickle_dev, model_name, resnet)
    import time
    for i in range(5):
        image = mio.import_builtin_asset('takeo.ppm', normalize=False)
        image = image.crop_to_landmarks_proportion(np.random.rand())
        start = time.time()
        boxes = DPMFitter.fast_fit_from_model(image, model, 0, return_once=True)
        stop = time.time()
        print('fitting tak:', stop-start)
        print('---------------------')

def create_response_map():
    images = mio.import_images('/vol/atlas/homes/jiankang/COFW_test', normalize=False)
    resnet = ResnetFeaturePyramid()
    for image in images:
        fea = resnet.extract_feature(image)
        cmap = plt.get_cmap('jet')
        sum_fea = np.sum(fea, axis=0) - fea[0, :, :]
        rgba_img = cmap(sum_fea)
        rgb_img = np.delete(rgba_img, 3, 2)
        rgb_img = np.transpose(rgb_img, [2, 0, 1])
        # response = response/np.max(response)
        # response = response/255.0
        print(rgba_img.shape)
        print(rgb_img.shape)
        response = Image(rgb_img)
        print('/vol/atlas/homes/jiankang/COFW_test/' + str(image.path).split('/')[-1].split('.')[0] + '_result.png')
        mio.export_image(response, '/vol/atlas/homes/ks3811/public/COFW_test/' + str(image.path).split('/')[-1].split('.')[0] + '_result.png', overwrite=True)
    print(images)

def find_hard_negatives():
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/5x5_dpm_filters'
    resnet = ResnetFeaturePyramid()
    dpm_learner = DPMLearner(feature_pyramid=resnet)
    model = dpm_learner.build_model(pickle_dev, 0)
    filters, defs2, defs, bias = mio.import_pickle('/vol/atlas/homes/ks3811/pickles/fast_dpm/5x5_dpm_filters.pkl')
    model.update_model_from_pickle(filters, defs2, defs, bias)

    hard_negatives = []
    _, negs = DPMLearner._get_frontal_pie_image_info(pickle_dev)
    for neg in negs:
        image = mio.import_image(neg['im'], normalize=False)
        boxes = DPMFitter.fast_fit_from_model(image, model, -2)
        print(boxes)
        if len(boxes) > 0:
            max_score = np.max(np.array([box['s'] for box in boxes]))
            hard_negatives.append({'neg': neg, 'max_score': max_score, 'boxes': boxes})
            if max_score >= -1:
                print(neg['im'], max_score)
    mio.export_pickle(hard_negatives, '/vol/atlas/homes/ks3811/pickles/resnet/5x5_dpm_filters/hard_negatives.pkl')
    print(len(hard_negatives))


def build_init_model():
    resnet = ResnetFeaturePyramid()
    # model = get_model(pickle_dev, model_name, resnet)
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/resnet/5x5_dpm_filters/pose_1'
    learner = DPMLearner(feature_pyramid=resnet)
    model = learner.build_model(pickle_dev, 1)
    pass

if __name__ == "__main__":
    # sys.exit(visualize_tree())
    # sys.exit(train_parts())
    # sys.exit(find_hard_negatives())
    # sys.exit(test_learn_model())
    # sys.exit(build_init_model())
    sys.exit(create_response_map())
