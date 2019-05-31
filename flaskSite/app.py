# import the Flask class from the flask module
from flask import Flask, render_template, request
from support.textFreq import recommendDoctor

# create the application object
app = Flask(__name__)

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('index.html')  # render a template


@app.route('/dentistRecommend',methods=['POST','GET'])
def dentistRecommend():

  
    topNRankNum = 5
    splitNum = 300
    contractDir = './company0/'

    if request.method == 'POST':
        #searchQuery = request.args.get('search',None)
        searchQuery = request.form['search']
        print(searchQuery)
    

    topNRankNum  = request.args.get('top',None)

    if topNRankNum is None:
        topNRankNum = 5

    




    resText,resRank, mainContract = recommendDoctor(topNRankNum,splitNum,contractDir)


    return render_template('dentistRecommend.html', **locals())


# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
