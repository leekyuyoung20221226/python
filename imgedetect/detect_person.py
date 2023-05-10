import numpy as np
import cv2
import winsound
classes = []
# 여기서 사용하는 파일은 
# https://drive.google.com/drive/folders/1rXdgki3gsB2Rsu6EV_eqNsaizqn8PZKn?usp=sharing
with open('./coco.names.txt','r') as f:
    classes = [line.strip() for line in f.readlines()]
    colors = np.random.uniform(0,255,size=(len(classes),3))
    # 모델로드
    yolo_model =  cv2.dnn.readNet('./yolov3.weights','./yolov3.cfg')
    layer_names =  yolo_model.getLayerNames()
    out_layers = [layer_names[i-1] for i in yolo_model.getUnconnectedOutLayers()]
    
# opencv와 yolo를 이용한 이미지 detecting
video = cv2.VideoCapture(0)
while video.isOpened():
    success,img =video.read()
    if success:
        height,width,channels = img.shape
        blob = cv2.dnn.blobFromImage(img,0.1/256, 
                                     (448,448),(0,0,0),swapRB=True,crop=False)
        yolo_model.setInput(blob) # 동영상을 yolo에 입력
        output3 = yolo_model.forward(out_layers)
        
        class_ids, confidences,boxes = [],[],[]    
        for output in output3:
          for vec85 in output:  # 85개의 요소를 가지고 있고 처음 5개는 바운딩 박스의 위치 , 나머지가 클래스에 대한 신뢰도
            scores = vec85[5:] # 확률이 0.5를 넘는 바운딩 박스를 모음
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # 신뢰도가 50%이상인 경우만
            if confidence > 0.5:
              centerx, centery = int(vec85[0]*width), int(vec85[1]*height)
              w,h = int(vec85[2]*width), int(vec85[3]*height)
              x,y = int(centerx-w/2), int(centery-h/2)
              boxes.append([x,y,w,h])
              confidences.append(float(confidence))
              class_ids.append(class_id)
          indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)  # 최대치의 box만남긴다
          # 살아남은 박스를 영상에 표시
          for i in range(len(boxes)):
              if i in indexes:
                x,y,w,h = boxes[i]
                text = f"{classes[class_ids[i]]} {confidences[i]:.3f}"
                cv2.rectangle(img, (x,y), (x+w, y+h), colors[class_ids[i]],2 )
                cv2.putText(img, text, (x,y+30), cv2.FONT_HERSHEY_PLAIN,
                            2,colors[class_ids[i]],2)
        cv2.imshow("detect human", img)
        if 0 in class_ids:
            print('detect person')
            winsound.Beep(2000, 500)
    key = cv2.waitKey(1) & 0xff
    if key==27: break  # 27은 esc 키
video.release()
cv2.destroyAllWindows()    
        
                
        
        
        
        
        
        
        
        
        
    
