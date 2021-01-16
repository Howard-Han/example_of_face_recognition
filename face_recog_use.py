from face_recog_func import *
import pandas as pd
import time


def do_training():
    print("模型訓練中...")
    X_1, y_1 = prepare_to_train("./train")
    model_log = train_logistic(X_1, y_1, model_save_path="./model/logistic.clf")
    model_knn = train_knn(X_1, y_1, model_save_path="./model/knn.clf")
    thres_log, max_acc_log = find_thres_log(X_1, y_1, "./model/logistic.clf")
    thres_knn, max_acc_knn = find_thres_knn(X_1, y_1, "./model/knn.clf")
    if max_acc_log>max_acc_knn:
        my_use, my_thres = 'logistic', thres_log
    else:
        my_use, my_thres = 'knn', thres_knn
    print("模型訓練完成，可執行特定人臉辨識!!")
    return my_use, my_thres

def do_predict(my_use, my_thres):
    # get 相簿的照片
    photo_path = "./photo"
    photo_list = os.listdir(photo_path)
    # prediction
    for photo_nowuse in photo_list:
        try:
            if my_use == 'logistic':
                pred, _ = predict_logistic(photo_path+"/"+photo_nowuse, model_path="./model/logistic.clf", prob_threshold=my_thres)
            else:
                pred, _ = predict_knn(photo_path+"/"+photo_nowuse, model_path="./model/knn.clf", distance_threshold=my_thres)
            # 辨別/講述是誰(二人以上要考慮方位)
            if len(pred)==0:
                print("前方沒有人臉以供辨識!!")
            elif len(pred)==1:
                if pred[0]=="unknown":
                    print("前方為未知的人，請小心!!")
                    # 畫框框(此處僅 display)
                    show_prediction_labels_on_image(photo_path+"/"+photo_nowuse, pred, (136,136,255), "display")
                else:
                    print("前方為{0}".format(pred[0]))
                    # 畫框框(此處僅 display)
                    show_prediction_labels_on_image(photo_path+"/"+photo_nowuse, pred, (136,136,255), "display")
            else:
                label_v, left_v = [i[0] for i in pred], [int(i[1][3]) for i in pred]
                for i in range(len(label_v)):
                    if label_v[i]=="unknown":
                        label_v[i] = "未知"
                mydict = {"label":label_v, "left":left_v}
                mydf = pd.DataFrame(mydict)
                mydf.sort_values(by="left", inplace=True)
                mydf.reset_index(inplace=True)
                label_align = [mydf.loc[i,"label"] for i in range(mydf.shape[0])]
                print("前方從左到右依序為: {0}".format(", ".join(label_align)))
                # 畫框框(此處僅 display)
                show_prediction_labels_on_image(photo_path+"/"+photo_nowuse, pred, (255,136,136), "display")
        except:
            print("預測圖片 {0} 時有錯誤發生!!!".format(photo_nowuse))


### start!!
s = input("請給予指示 (1 = 模型訓練, 2 = 預測圖片, 3 = 離開) : ")
time.sleep(1)
while 1<1000:
    if s=="1":
        my_use, my_thres = do_training()
        time.sleep(1)
        s = input("請給予指示 (1 = 模型訓練, 2 = 預測圖片, 3 = 離開) : ")
    elif s=="2":
        try:
            do_predict(my_use, my_thres)
            time.sleep(1)
            s = input("請給予指示 (1 = 模型訓練, 2 = 預測圖片, 3 = 離開) : ")
        except:
            print("有錯誤發生!!!")
            time.sleep(1)
            s = input("請給予指示 (1 = 模型訓練, 2 = 預測圖片, 3 = 離開) : ")
    elif s=="3":
        break
    else:
        print("指示不符規定!!!")
        time.sleep(1)
        s = input("請給予指示 (1 = 模型訓練, 2 = 預測圖片, 3 = 離開) : ")


