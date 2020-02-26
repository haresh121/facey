import os
import pickle
import matplotlib.pyplot as plt
import cv2
import face_recognition as FR

# K.set_image_dim_ordering('th')


class Model:
    """
        hello
    """

    def __init__(self):
        """
            hello
        """
        self.faceCascade = cv2.CascadeClassifier(
            "./haarcascade_frontalface_default.xml"
        )
        print("Model initialized for ", len(os.listdir("./data")), " members")

    def save_data(self):
        """
        hello
        """
        faces = []
        data = os.listdir("./data")
        for i in range(len(data)):
            data[i] = "./data/" + data[i] + "/1.jpg"
        for i in data:
            face = FR.load_image_file(i)
            face = FR.face_encodings(face)[0]
            faces.append(face)
        pick = open("./fdata/fdata.pickle", "wb+")
        pickle.dump(faces, pick)

    def load_and_predict(self, img):
        """
            hello
        """
        pick = open("./fdata/fdata.pickle", "rb")
        faces = pickle.load(pick)
        faces_names = os.listdir("./data")
        try:
            uenc = FR.face_encodings(img)[0]
        except IndexError:
            name = "Unknown"
        else:
            name = FR.compare_faces(faces, uenc)
            for i in range(len(faces)):
                if name[i]:
                    break
            name = faces_names[i]
        return name

    def predict_from_cam(self, cam=True, pic=False, im=None):
        """
            hello
        """
        if cam and pic and im is not None:
            cam = False
        elif cam and pic and im is None:
            pic = False
        if pic and im is None:
            print(b"No image provided, please provide a image")
        font = cv2.FONT_HERSHEY_PLAIN
        if cam:
            cap = cv2.VideoCapture(0)
            while True:
                # j+=1
                _, frame = cap.read()
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                try:
                    faces = self.faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=5,
                        minSize=(30, 30),
                        flags=cv2.CASCADE_SCALE_IMAGE,
                    )
                except IndexError:
                    faces = None
                else:
                    for (x, y, w, h) in faces:
                        # cv2.rectangle(frame, (x, y), (x+w, y+h),
                        #               (255, 0, 0), 3)
                        roi_color = frame[y : y + h, x : x + w]
                        name = self.load_and_predict(roi_color)
                        cv2.rectangle(
                            frame, (x, y), (x + w + 50, y + h + 50), (0, 255, 0), 1
                        )
                        # print(name)
                        cv2.putText(
                            frame, name, (x - 10, y - 10), font, 1, (0, 0, 255), 1
                        )
                cv2.imshow("Face", frame)
            cap.release()
            cv2.destroyAllWindows()
        if pic:
            names = []
            frame = plt.imread(im)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
            except IndexError:
                faces = None
            else:
                for (x, y, w, h) in faces:
                    # cv2.rectangle(frame, (x, y), (x+w, y+h),
                    #               (255, 0, 0), 3)
                    roi_color = frame[y : y + h, x : x + w]
                    name = self.load_and_predict(roi_color)
                    names.append(name)
            return names

    def addToDb(self):
        """
        hello
        """
        name = input("Enter Name:")
        base_valid = "./data/"
        try:
            os.mkdir(base_valid + name)
        except FileExistsError:
            pass
        cap = cv2.VideoCapture(0)
        j = 0
        while True:
            _, frame = cap.read()
            for i in range(1):
                j += 1
                imgname = base_valid + name + "/" + str(i + 1) + ".jpg"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE,
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                    roi_color = frame[y : y + h, x : x + w]
                    cv2.imwrite(imgname, roi_color)
                cv2.imshow("Image", frame)
            if (cv2.waitKey(30) & 0xFF == ord("q")) or j == 2:
                break
        cap.release()
        cv2.destroyAllWindows()
        self.save_data()
