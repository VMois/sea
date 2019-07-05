import os
import shutil
import datetime
import matplotlib.pyplot as plt


def save_training(model, history):
    os.makedirs('models', exist_ok=True)
    training_name = datetime.datetime.now().strftime(f'%Y-%m-%d_%H:%M:%S')
    model_folder_path = os.path.join('models', training_name)
    shutil.rmtree(model_folder_path, ignore_errors=True)
    os.makedirs(model_folder_path)

    # save model weights
    model.save(os.path.join(model_folder_path, 'weights.h5'))

    # save model
    model_json = model.to_json()
    with open(os.path.join(model_folder_path, 'model.json'), 'w') as json_file:
        json_file.write(model_json)

    # save plot
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{training_name} model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_folder_path, 'loss_curve.png'))

    # save accuracy
    plt.clf()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title(f'{training_name} model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(os.path.join(model_folder_path, 'accuracy_curve.png'))
