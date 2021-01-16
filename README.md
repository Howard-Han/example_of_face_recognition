# example_of_face_recognition

### 我用 Windows + Python 3.8.2 操作的，其他作業系統沒測試過
### 關於 face_recognition 套件的安裝請詳見 https://github.com/ageitgey/face_recognition

### 所需python套件:
1. numpy
2. pandas
3. PIL (pip install pillow)
4. sklearn (pip install scikit-learn)
5. face_recognition

### 操作步驟:
1. 下載所有的資料夾與py檔
2. 將 train 裡面的資料夾改換成你想要辨識的人的個人照片(每個人都需有一個資料夾，建議至少2人且每人至少2張照片)
3. 將想測試的照片放入 photo
4. 開啟命令提示字元(cmd)並執行 face_recog_use.py
5. 在cmd中，輸入1以訓練模型，輸入2以預測圖片，輸入3以離開程式

### 備註:
1. 圖片僅支援 png, jpg, jpeg 三種格式
2. 此處我只用羅吉斯迴歸和KNN做監督式學習，事實上可以加入其他方法一起比較(ex. 決策樹、SVM、貝氏分類器、...)
3. 在 face_recog_func.py 中，我有保留「比較模型」的code，是之前我用四種監督式學習方法的比較，有興趣的朋友可以拿去改成自己想用的方法
4. show_prediction_labels_on_image 函式一共有5個參數，分別為圖片路徑、圖片預測結果(包含標籤與四點)、框框顏色(RGB)、狀態(預設為"display")、結果儲存路徑(預設為None)。狀態如果是預設值就僅作展示，反之則會將結果儲存成圖片(結果儲存路徑需不為None)
5. 操作步驟5中，無法在未輸入過1之前先輸入2 (會跳出Error)

# Thanks
***
> Thanks for ageitgey's code !!   Link is here: https://github.com/ageitgey/face_recognition
***
> photo creditted by 木曜4超玩 (https://www.youtube.com/channel/UCLW_SzI9txZvtOFTPDswxqg)
