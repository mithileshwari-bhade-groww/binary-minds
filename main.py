import pickle  # If your model is serialized with pickle
from flask import Flask, request, jsonify
import pandas as pd
# Create a Flask app
app = Flask(__name__)


# Load your trained machine learning model
# Replace 'model.pkl' with your actual model file
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Define a route for your API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        result = request.json
        print(result)
        # Convert the dictionary to a DataFrame
        df = pd.DataFrame([result])  # Assuming the response is a single JSON object
        #

        # # Display the DataFrame
        print(df)
# data['contractSymbol']
#         data['buySell']
        # Perform inference using your ML model
        modelInput = {
            "index" : 1,
            "Call_put": "CALL",
            "buy_sell": "B",
            "gb" : 17171.11,
            "strike" : 171.11,
            "days_to_expiry": 1,

        }
        df_test = pd.DataFrame([modelInput])
        df_test['UID'] = 'aab' +'_'+ df_test['Call_put'] +'_'+ df_test['buy_sell'] +'_'+ df_test['days_to_expiry'].astype(str)

        # print(df_test)
        one_hot = pd.get_dummies(df_test)
        resultML = model.predict(one_hot)
        print(round(resultML[0]))
        # result = [1, 2, 3, 4, 5]
        # print(modelInput)
        # # You can also post-process the result here if needed
        #
        # # Return the result as JSON
        response = {
            "qty": round(resultML[0]),
            "orderType": 'MKT',
            "productType": "NRML",
            "advanceOrderType": 0,
            "isUseMaxBuyQty": False,

        }
        # print(response)
        return response
        # return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
    # app.run(host='0.0.0.0', port=5000)
