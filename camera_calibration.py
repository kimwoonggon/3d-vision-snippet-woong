import numpy as np
import cv2 as cv
import glob
import os 
import matplotlib.pyplot as plt
import time

def capture_images(interval=7, num_images=2, save_path='demoImages/calibration/woong/distorted'):
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    cap = cv.VideoCapture(0)  # 0 is typically the default camera
    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    print("Starting image capture...")
    try:
        for i in range(num_images):
            ret, frame = cap.read()
            if ret:
                timestamp = int(time.time())
                img_filename = os.path.join(save_path, f'img_{timestamp}.jpg')
                cv.imwrite(img_filename, frame)
                print(f"Captured {img_filename}")
                time.sleep(interval)  # Wait for specified interval (5 seconds)
            else:
                print("Error: Frame capture failed")
                break
    finally:
        cap.release()

def calibrate(showPics=True):
    root = os.getcwd()
    calibrationDir = os.path.join(root,'demoImages//calibration//woong')
    imgPathList = glob.glob(os.path.join(calibrationDir,'*.jpg'))

    nRows = 10 
    nCols =  7
    termCriteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,30,0.001)
    worldPtsCur = np.zeros((nRows*nCols,3), np.float32)
    worldPtsCur[:,:2] = np.mgrid[0:nRows,0:nCols].T.reshape(-1,2)
    worldPtsList = []
    imgPtsList = [] 
    print(imgPathList)
    for curImgPath in imgPathList:
        imgBGR = cv.imread(curImgPath)
        imgGray = cv.cvtColor(imgBGR, cv.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv.findChessboardCorners(imgGray,(nRows,nCols), None)

        if cornersFound == True:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv.cornerSubPix(imgGray,cornersOrg,(11,11),(-1,-1),termCriteria)
            imgPtsList.append(cornersRefined)

            if showPics: 
                print(showPics)
                cv.drawChessboardCorners(imgBGR,(nRows,nCols),cornersRefined,cornersFound)
                cv.imshow('Chessboard', imgBGR)
                cv.waitKey(0)
        else:
            print("Corners not found in image: ", curImgPath)
    cv.destroyAllWindows()
    if not worldPtsList or not imgPtsList:
        raise ValueError("No corners were found in any images.")
    repError,camMatrix,distCoeff,rvecs,tvecs = cv.calibrateCamera(worldPtsList, imgPtsList, imgGray.shape[::-1],None,None)
    print('Camera Matrix:\n',camMatrix)
    print("Reproj Error (pixels): {:.4f}".format(repError))
    
    curFolder = os.path.dirname(os.path.abspath(__file__))
    paramPath = os.path.join(curFolder,'calibration.npz')
    np.savez(paramPath, 
        repError=repError, 
        camMatrix=camMatrix, 
        distCoeff=distCoeff, 
        rvecs=rvecs, 
        tvecs=tvecs)
    
    return camMatrix,distCoeff

def removeDistortion(camMatrix,distCoeff): 
    root = os.getcwd()
    imgPath = os.path.join(root,'demoImages//calibration//woong//distorted//img_1713892210.jpg')
    img = cv.imread(imgPath)
    print(camMatrix, distCoeff)
    height,width = img.shape[:2]
    camMatrixNew,roi = cv.getOptimalNewCameraMatrix(camMatrix,distCoeff,(width,height),1,(width,height)) 
    imgUndist = cv.undistort(img,camMatrix,distCoeff,None,camMatrixNew)

    #cv.line(img,(1769,103),(1780,922),(255,255,255),2)
    #cv.line(imgUndist,(1769,103),(1780,922),(255,255,255),2)

    plt.figure() 
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(imgUndist)
    plt.show()

def runCalibration(): 
    calibrate(showPics=True) 

def runRemoveDistortion(): 
    camMatrix,distCoeff = calibrate(showPics=False) 
    removeDistortion(camMatrix,distCoeff)

if __name__ == '__main__': 
    #runCalibration() 
    runRemoveDistortion()
    #capture_images()
