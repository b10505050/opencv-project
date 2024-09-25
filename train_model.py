import cv2
import os
import numpy as np
import json

def train_model(data_path, model_save_path, label_dict_path):
    #引入LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
   
    #初始化label以存取方便之後分類data
    faces, labels = [], []
    label_dict = {}

    #找/data 下的各個資料夾，以此為例只有Ko
    for label, person_name in enumerate(os.listdir(data_path)):
        person_dir = os.path.join(data_path, person_name)
        # 不是目錄就跳過
        if not os.path.isdir(person_dir):
            continue
        # 將標籤和人名添加到預留的label
        label_dict[label] = person_name
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            #將圖片轉為灰度
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            faces.append(image)
            labels.append(label)
    
    #train人臉識別模型
    face_recognizer.train(faces, np.array(labels))
    #保存模型
    face_recognizer.save(model_save_path)
    #存取標籤
    with open(label_dict_path, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False)
    return label_dict

if __name__ == "__main__":
    data_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/data/known_faces'
    model_save_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/models/trained_face_recognition_model.yml'
    label_dict_path = '/home/jimmy/camera-opencv/camera-opencv/03-face_datection/models/label_dict.json'
    label_dict = train_model(data_path, model_save_path, label_dict_path)
    print("Model trained and saved at", model_save_path)
    print("Label dictionary saved at", label_dict_path)
    print("Label dictionary:", label_dict)
