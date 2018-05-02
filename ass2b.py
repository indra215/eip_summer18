import numpy as np

def sigmoid(inp_data):
	sig_inp_data = list(map(lambda x: 1/(1+np.exp(-x)), inp_data))
	sig_inp_data = np.array(sig_inp_data)
	return sig_inp_data

def derivatives_sigmoid(inp_data):
	dsigmoid = list(map(lambda x: x*(1-x), inp_data))
	dsigmoid = np.array(dsigmoid)
	return dsigmoid

if __name__ == '__main__':

	''' step0 '''
	X = np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
	Y = np.array([[1],[1],[0]])
	
	''' step1 '''
	wh = np.random.rand(4,3)
	bh = np.random.rand(1,3)
	wout = np.random.rand(3,1)
	bout = np.random.rand(1,1)
	print(wh)
	print(bh)
	print(wout)
	print(bout)
	
	''' step2 '''
	hidden_layer_input = np.dot(X, wh) + bh
	print(hidden_layer_input)

	''' step3 '''
	hidden_layer_activations = sigmoid(hidden_layer_input)
	print(hidden_layer_activations)

	''' step4 '''
	output_layer_input = np.dot(hidden_layer_activations, wout) + bout
	output = sigmoid(output_layer_input)
	print(output)

	''' step5 '''
	E = Y - output
	print(E)

	''' step6 '''
	slope_output_layer = derivatives_sigmoid(output)
	slope_hidden_layer = derivatives_sigmoid(hidden_layer_activations)
	print(slope_hidden_layer)
	print(slope_output_layer)

	''' step7 '''
	lr = 0.1
	d_output = E * slope_output_layer * lr
	print(d_output)

	''' step8 '''
	error_at_hidden_layer = np.dot(d_output, np.transpose(wout))
	print(error_at_hidden_layer)

	''' step9 '''
	d_hiddenlayer = error_at_hidden_layer * slope_hidden_layer
	print(d_hiddenlayer)

	''' step10 '''
	wout = wout + (np.dot(np.transpose(hidden_layer_activations), d_output)* lr )
	print(wout)
	wh = wh + (np.dot(np.transpose(X), d_hiddenlayer) * lr)
	print(wh)
	
	''' step11 '''
	bout = bout + (np.sum(d_output, axis=0) * lr)
	print(bout)
	bh = bh + (np.sum(d_hiddenlayer, axis=0) * lr)
	print(bh)