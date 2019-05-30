# import the Flask class from the flask module
from flask import Flask, render_template, request
from support.textFreq import textFreqCal

# create the application object
app = Flask(__name__)

# use decorators to link the function to a url
@app.route('/')
def home():
    return render_template('index.html')  # render a template


@app.route('/termsService',methods=['POST','GET'])
def termsService():

    topNRankNum = 5
    splitNum = 300
    contractDir = './company0/'

    if request.method == 'POST':
        result = request.form
        if result['top'].isdigit():
            topNRankNum = int(result['top'])
        if result['contractType'] == 'gym':
            splitNum = 300
            contractDir = './company0/'
        elif result['contractType'] == 'dating':
            splitNum = 400
            contractDir = './dating/'
        else:
            # We are doing hotel
            splitNum = 400
            contractDir = './hotel/'
    else:
        topNRankNum  = request.args.get('top',None)

    if topNRankNum is None:
        topNRankNum = 5

    




    resText,resRank, mainContract = textFreqCal(topNRankNum,splitNum,contractDir)


    return render_template('termsOfService.html', **locals())


# start the server with the 'run()' method
if __name__ == '__main__':
    app.run(debug=True)
