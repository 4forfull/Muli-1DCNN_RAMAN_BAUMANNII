import numpy as np
from pandas import read_csv, DataFrame
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def CNN_pre(filepath):
    print('Start analysis, Please Wait ... ...')
    df = read_csv(filepath)
    df1 = df.loc[:, df.columns != 'label']
    df1 = df1.T
    # df1 = MinMaxScaler().fit_transform(df1)
    df1 = DataFrame(df1)
    df = df1.T

    X = np.expand_dims(df.values[:, 0:1160].astype(float), axis=2)

    json_file = open(r"Multiscale-CNN.json", "r")
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("Multiscale-CNN.h5")

    # 输出预测类别
    predicted_label = loaded_model.predict(X)
    predicted_label = np.argmax(predicted_label, axis=1)
    predicted_pro = loaded_model.predict(X)
    predicted_pro = np.max(predicted_pro)

    if predicted_label == 0:
        print("The Analysis Result is: ST1828, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 1:
        print("The Analysis Result is: ST1433, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 2:
        print("The Analysis Result is: ST1333, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 3:
        print("The Analysis Result is: ST1276, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 4:
        print("The Analysis Result is: ST1264, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 5:
        print("The Analysis Result is: ST1159, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 6:
        print("The Analysis Result is: ST833, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 7:
        print("The Analysis Result is: ST821, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 8:
        print("The Analysis Result is: ST795, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 9:
        print("The Analysis Result is: ST768, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 10:
        print("The Analysis Result is: ST629, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 11:
        print("The Analysis Result is: ST516, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 12:
        print("The Analysis Result is: ST457, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 13:
        print("The Analysis Result is: ST433, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 14:
        print("The Analysis Result is: ST410, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 15:
        print("The Analysis Result is: ST396, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 16:
        print("The Analysis Result is: ST357, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 17:
        print("The Analysis Result is: ST338, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 18:
        print("The Analysis Result is: ST336, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 19:
        print("The Analysis Result is: ST321, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 20:
        print("The Analysis Result is: ST221, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 21:
        print("The Analysis Result is: ST220, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 22:
        print("The Analysis Result is: ST217, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 23:
        print("The Analysis Result is: ST204, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 24:
        print("The Analysis Result is: ST203, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 25:
        print("The Analysis Result is: ST132, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 26:
        print("The Analysis Result is: ST119, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 27:
        print("The Analysis Result is: ST106, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 28:
        print("The Analysis Result is: ST77, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 29:
        print("The Analysis Result is: ST71, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 30:
        print("The Analysis Result is: ST68, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 31:
        print("The Analysis Result is: ST64, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 32:
        print("The Analysis Result is: ST63, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 33:
        print("The Analysis Result is: ST52, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 34:
        print("The Analysis Result is: ST46, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 35:
        print("The Analysis Result is: ST40, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 36:
        print("The Analysis Result is: ST33, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 37:
        print("The Analysis Result is: ST25, Predicted Probability = {:.2%}".format(predicted_pro))

    elif predicted_label == 38:
        print("The Analysis Result is: ST10, Predicted Probability = {:.2%}".format(predicted_pro))

    else:
        print("The Analysis Result is:  ST2, Predicted Probability = {:.2%}".format(predicted_pro))


if __name__ == "__main__":
    file_path = 'D:/拉曼光谱/鲍曼不动杆菌/全部数据/demo10.csv'
    CNN_pre(file_path)