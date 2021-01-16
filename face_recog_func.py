import math
import os
import os.path
import pickle
from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import numpy as np
# sklearn
from sklearn import linear_model, neighbors # 此處我只用羅吉斯迴歸和KNN做監督式學習，事實上可以加入其他方法一起比較
from sklearn import metrics

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


## 0. data prepare
def prepare_to_train(train_dir):
    X, y = [], []
    # 迴圈 by train_dir 內的每個人
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue
        # 迴圈 by 每個人資料夾內的所有訓練用照片
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)
            if len(face_bounding_boxes) != 1:
                # 如果在該訓練用照片中沒有人(或是太多人)，略過它
                if verbose:
                    print("照片 {} 不適合做訓練: {}".format(img_path, "找不到臉" if len(face_bounding_boxes) < 1 else "找到超過一張臉"))
            else:
                # 在訓練資料集中加入該照片之 face encoding
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)
                print("照片 {} 適合做訓練".format(img_path))
    return X, y


## 1. logistic reg
def train_logistic(X_train, y_train, model_save_path=None, verbose=False):
    # Create and train the classifier
    model_ovr = linear_model.LogisticRegression(penalty="none", solver="saga", max_iter=10000, multi_class="ovr")
    model_mul = linear_model.LogisticRegression(penalty="none", solver="saga", max_iter=10000, multi_class="multinomial")
    fit_ovr = model_ovr.fit(X_train, y_train)
    fit_mul = model_mul.fit(X_train, y_train)
    acc_ovr = model_ovr.score(X_train, y_train)
    acc_mul = model_mul.score(X_train, y_train)
    if verbose:
        if acc_ovr>acc_mul:
            print("Choose One-vs-Rest method for trainiing, accuracy = {0}".format(acc_ovr))
        else:
            print("Choose multinomial method for trainiing, accuracy = {0}".format(acc_mul))
    my_model = model_ovr if acc_ovr>acc_mul else model_mul
    # Save the trained classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(my_model, f)
    return my_model

def predict_logistic(X_img_path, clf=None, model_path=None, prob_threshold=0.75):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if clf is None and model_path is None:
        raise Exception("Must supply classifier either thourgh clf or model_path")
    # 載入以訓練之模型 (clf 或 model_path 至少一個非None)
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
    # 載入目標照片並找出臉的位置
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    # 如果該照片中沒有任何臉，return empty lists.
    if len(X_face_locations) == 0:
        return [], []
    else:
        # 取得各臉之 face encoding
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
        # 運用模型找到各臉之最佳 match
        y_pred, y_prob = clf.predict(faces_encodings), clf.predict_proba(faces_encodings)
        are_matches = [np.any(y_prob[i]>prob_threshold) for i in range(len(X_face_locations))]
        # (未達門檻值的定為 unknown)
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(y_pred, X_face_locations, are_matches)], y_prob


## 2. KNN
def train_knn(X_train, y_train, model_save_path=None):
    # Create and train the classifier
    k = int(round(math.sqrt(len(X_train))))
    my_model = neighbors.KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', weights='distance')
    my_model.fit(X_train, y_train)
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(my_model, f)
    return my_model

def predict_knn(X_img_path, clf=None, model_path=None, distance_threshold=0.6):
    if not os.path.isfile(X_img_path) or os.path.splitext(X_img_path)[1][1:] not in ALLOWED_EXTENSIONS:
        raise Exception("Invalid image path: {}".format(X_img_path))
    if clf is None and model_path is None:
        raise Exception("Must supply classifier either thourgh clf or model_path")
    # 載入以訓練之模型 (clf 或 model_path 至少一個非None)
    if clf is None:
        with open(model_path, 'rb') as f:
            clf = pickle.load(f)
    # 載入目標照片並找出臉的位置
    X_img = face_recognition.load_image_file(X_img_path)
    X_face_locations = face_recognition.face_locations(X_img)
    # 如果該照片中沒有任何臉，return empty lists.
    if len(X_face_locations) == 0:
        return [], []
    else:
        # 取得各臉之 face encoding
        faces_encodings = face_recognition.face_encodings(X_img, known_face_locations=X_face_locations)
        # 運用模型找到各臉之最佳 match
        closest_distances = clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]
        y_prob = clf.predict_proba(faces_encodings)
        # (未達門檻值的定為 unknown)
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(clf.predict(faces_encodings), X_face_locations, are_matches)], y_prob


'''
## note: 比較模型
def matrix_trans(mylist):
    n = len(mylist[0])
    my_matrix = np.array(mylist[0]).reshape(1,n)
    for i in range(1, len(mylist)):
        nowuse = np.array(mylist[i]).reshape(1,n)
        my_matrix = np.vstack((my_matrix,nowuse))
    return my_matrix
def four_model_test(test_dir):
    test_log, test_lda, test_knn, test_svm = [], [], [], []
    prob_log, prob_lda, prob_knn, prob_svm = [], [], [], []
    for image_file in os.listdir(test_dir):
        full_file_path = os.path.join(test_dir, image_file)
        #print("??   ",full_file_path)
        pred_log, prob_1 = predict_logistic(full_file_path, model_path="D:/face_recognition_folder/model_210113/logistic.clf")
        test_log.append(pred_log[0][0])
        prob_log.append(prob_1[0])
        pred_lda, prob_2 = predict_LDA_QDA(full_file_path, model_path="D:/face_recognition_folder/model_210113/lda_qda.clf")
        test_lda.append(pred_lda[0][0])
        prob_lda.append(prob_2[0])
        pred_knn, prob_3 = predict_knn(full_file_path, model_path="D:/face_recognition_folder/model_210113/knn.clf")
        test_knn.append(pred_knn[0][0])
        prob_knn.append(prob_3[0])
        pred_svm, prob_4 = predict_svm(full_file_path, model_path="D:/face_recognition_folder/model_210113/svm.clf")
        test_svm.append(pred_svm[0][0])
        prob_svm.append(prob_4[0])
    prob_log = matrix_trans(prob_log)
    prob_lda = matrix_trans(prob_lda)
    prob_knn = matrix_trans(prob_knn)
    prob_svm = matrix_trans(prob_svm)
    return [test_log, test_lda, test_knn, test_svm], [prob_log, prob_lda, prob_knn, prob_svm]
def calc_score_1(label_true, label_predict):
    return [metrics.accuracy_score(label_true, label_predict), metrics.f1_score(label_true, label_predict, average='macro')]
'''


## 3. 找出門檻值
def find_thres_log(X, y, model_path):
    thres_v, acc_v = [0.6,0.65,0.7,0.75,0.8,0.85,0.9], []
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    for t in thres_v:
        y_pred = []
        for X_one in X:
            X_a = np.array(X_one).reshape(1,len(X_one))
            pred_one, prob_one = clf.predict(X_a), clf.predict_proba(X_a)
            is_match = np.any(prob_one>t)
            y_pred.append(pred_one if is_match else "unknown")
        acc_v.append(metrics.accuracy_score(y, y_pred))
    check_v = [True if ii==max(acc_v) else False for ii in acc_v]
    return thres_v[np.where(check_v)[0][0]], max(acc_v)
def find_thres_knn(X, y, model_path):
    thres_v, acc_v = [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8], []
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    for t in thres_v:
        y_pred = []
        for X_one in X:
            X_a = np.array(X_one).reshape(1,len(X_one))
            closest_distances = clf.kneighbors(X_a, n_neighbors=1)
            is_match = closest_distances[0][0][0] <= t
            pred_one = clf.predict(X_a)
            y_pred.append(pred_one if is_match else "unknown")
        acc_v.append(metrics.accuracy_score(y, y_pred))
    check_v = [True if ii==max(acc_v) else False for ii in acc_v]
    return thres_v[np.where(check_v)[0][0]], max(acc_v)

## 4. 畫框框
def show_prediction_labels_on_image(img_path, predictions, color_t, status, path_for_save=None):
    pil_image = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(pil_image)
    for name, (top, right, bottom, left) in predictions:
        draw.rectangle(((left, top), (right, bottom)), outline=color_t)
        name = name.encode("UTF-8")
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=color_t, outline=color_t)
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
    # Remove the drawing library from memory as per the Pillow docs
    del draw
    # Display or save the resulting image
    if status=="display":
        pil_image.show()
    else:
        pil_image.save(path_for_save)


##
