# Import Libraries.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Build model.
def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(150,activation='relu'))
    model.add(Dense(120,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model