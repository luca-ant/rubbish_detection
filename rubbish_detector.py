from dataset import Dataset
class RubbishDetector():

    __instance = None

    @staticmethod
    def get_instance():

        if RubbishDetector.__instance == None:
            RubbishDetector("./")
        return RubbishDetector.__instance

    def __init__(self, working_dir):
        # private constructor
        if RubbishDetector.__instance != None:
            raise Exception("RubbishDetector class is a singleton! Use RubbishDetector.get_instance()")
        else:
            self.working_dir = working_dir
            self.model_dir = self.working_dir + "model/"
            self.weights_dir = self.working_dir + "weights/"
            self.train_dir= self.working_dir + 'training/'

            self.weights_file = self.weights_dir + "weights.h5"
            self.model_file = self.model_dir + "model.h5"
            self.last_epoch_file = self.train_dir + "last_epoch.txt"
            self.total_epoch_file = self.train_dir + "total_epoch.txt"

            self.dataset = Dataset(self.working_dir)
            self.last_epoch = 0
            self.batch_size = 16
            self.total_epochs = 50
            self.train_images = []
            self.test_images = []
            self.val_images = []

            self.model = None       

            RubbishDetector.__instance = self

    def create_nn(self, input_shape):
        print("CREATING MODEL")
        model = Sequential()
        input_image = Input(shape=input_shape)
        model.add(Dropout(0.5)(input_image))
        model.add(Dense(500, activation='relu', input_dim=8))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(50, activation='relu'))
        model.add(Dense(2, activation='softmax'))

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.model = model