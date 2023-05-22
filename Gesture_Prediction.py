import subprocess
import cv2
import numpy as np
import mediapipe as mp
from threading import Thread


from keras import load_model


import tensorflow as tf
#from tensorflow.keras.models import load_model
import time

def tv_call(tv_name):
    print("Entered GASDK:")
    if(int(tv_name)==0):
        subprocess.call(['python', '/home/akshit/env/lib/python3.10/site-packages/googlesamples/assistant/grpc/textinput.py', '--query-opencv', '0'])
    elif(int(tv_name)==1):
        subprocess.call(['python', '/home/akshit/env/lib/python3.10/site-packages/googlesamples/assistant/grpc/textinput.py', '--query-opencv', '1'])
    



def start():
    tv_state="k"
        # initialize mediapipe
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils

    # Load the gesture recognizer model
    model = load_model('mp_hand_gesture')

    # Load class names
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)


    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    temp=0
    while True:
        # Read each frame from the webcam
        _, frame = cap.read()

        x, y, c = frame.shape

        # Flip the frame vertically
        frame = cv2.flip(frame, 1)
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Get hand landmark prediction
        result = hands.process(framergb)

        # print(result)
        
        className = ''

        # post process the result
        if result.multi_hand_landmarks:
            landmarks = []
            for handslms in result.multi_hand_landmarks:
                for lm in handslms.landmark:
                    # print(id, lm)
                    lmx = int(lm.x * x)
                    lmy = int(lm.y * y)

                    landmarks.append([lmx, lmy])

                # Drawing landmarks on frames
                mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                # Predict gesture
                prediction = model.predict([landmarks])
                #print(prediction[0])
                # print(prediction)
                classID = np.argmax(prediction)
                className = classNames[classID]
                print("CLassNAME: "+className+"\nTv state: "+tv_state)
                

                if(temp==0 and className=="ON"):
                    tv_state="ON"
                    tv_name=0
                    thread = Thread(target=tv_call, args=(tv_name,))
                    temp=1
                    thread.start()
                elif(temp==0 and className=="OFF"):
                    tv_state="OFF"
                    tv_name=1
                    thread = Thread(target=tv_call, args=(tv_name,))

                    temp=1                
                    thread.start()
                if(className=="ON" and tv_state=="OFF"):
                    tv_state="ON"
                    tv_name=0
                    thread = Thread(target=tv_call, args=(tv_name,))

                    thread.start()
                    
                elif(className=="OFF" and tv_state=="ON"):
                    tv_state="OFF"
                    tv_name=1
                    thread = Thread(target=tv_call, args=(tv_name,))

                    thread.start()
                
        #time.sleep(0.5)


                

        # show the prediction on the frame
        cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0,0,255), 2, cv2.LINE_AA)

        # Show the final output
        cv2.imshow("Output", frame) 

        if cv2.waitKey(1) == ord('q'):
            break

    # release the webcam and destroy all active windows
    cap.release()

    cv2.destroyAllWindows()



start()