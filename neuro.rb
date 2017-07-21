require 'matrix'

# Matrix element wise matrix multiplication 
# Hadamard product (matrices):
# https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
def element_multiplication(m1, m2)
	m3 = Matrix.build(m1.row_count, m1.column_count) {|r, c| m1[r, c] * m2[r, c]}
	return m3
end

# Summation of all values in a matrix
def element_summation(m)
	s = 0
	m.each {|x| s += x}
	return s
end

# a confusion matrix illustrates the accuracy of a classifier:
# values in the diagonal of the classifier are correctly classified.
# https://en.wikipedia.org/wiki/Confusion_matrix
def confusion_matrix(expected, predicted)
	expected = expected.to_a.map {|x| x.index(x.max)}
	predicted = predicted.to_a.map {|x| x.index(x.max)}
	
	n = (expected + predicted).uniq.length
	cm = Matrix.build(n){0}.to_a
	expected.zip(predicted).map {|x, y| cm[x][y]+=1}
	
	return Matrix.rows(cm)
end

#The actual neural network
class NeuralNetwork
	def initialize()
		# For the sake of simplicity feel free to hardcode
		# parameters. The goal is a working feedforward neral
		# network. One hidden layer, one input layer, and one
		# output layer are enough to achieve 99% accuracy on
		# the data set.
	end

	##############################################
	def train(x, y)
		# the training method that updates the internal weights
		# using the predict
	end

	##############################################
	def predict(x)
	end

	##############################################
	protected

		##############################################
		def propagate(x)
			# applies the input to the network
			# this is the forward propagation step
		end

		##############################################
		def back_propagate(x, y, y_hat)
			# goes backwards and finds the weights
			# that need to be tuned
		end
end
