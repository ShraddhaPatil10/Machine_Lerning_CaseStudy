from sklearn import tree

#rough=1
#smooth=0

#Tennis=1
#Cricket=2

def BallPredictor(weight,surface):
    Features=[[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]

    Labels=[1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    #Decide the machine learning algorithm
    obj = tree.DecisionTreeClassifier()

    #Perform the training of model
    obj = obj.fit(Features,Labels)

    #Perform the testing
    Ret = obj.predict([[weight,surface]])

    if Ret==1:
        print("Your object looks like a tennis ball")

    else:
        print("Your object looks like a cricket ball")

def main():
    print("----------------------Ball Predictor case study------------------")
    print("Please enter your weight of ball in grams:")
    weight=int(input())

    print("Please enter the surface of your object either(smooth/rough)")
    surface=input()

    if surface.lower()=="rough":
        surface=1

    elif surface.lower()=="smooth":
        surface=0

    else:
        print("Invalid type of surface")
        exit()

    BallPredictor(weight,surface)

if __name__=="__main__":
    main()