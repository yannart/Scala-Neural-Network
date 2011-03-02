/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package com.yannart.neuralnetwork

import scala.collection.mutable.{ListBuffer, StringBuilder}
import scala.math.{exp, pow, random, sqrt}
import scala.xml.{Elem, XML}


abstract class ActivationFunction{
	def eval( value : Double) : Double
} 

object Sigmoid extends ActivationFunction {
	override def eval ( value: Double) : Double = {
		1d / (1d + exp(-value))
	}
}

class Neuron (val weights: Array[Double]){
	
	import Sigmoid.eval
	
	val pastErrors = new Array[Double](weights.size)
	
	def this (inputNum: Int) = {
	
		//Use [-2.4, 2.4] as initial weight range
		this(Array.fill(inputNum)(random * (2.4 *2)- 2.4))
	}
	
	def run(inputs: Array[Double]) : Double = {

		//TODO: More functional style
		var cumul : Double = 0d
		for(i <- 0 until inputs.size){
			cumul += inputs(i) * weights(i)
		}
		eval(cumul)
	}
	
	override def toString() : String = {
		
		var buf = new StringBuilder("\n\t(")
		for(i <- weights.indices){
			 buf.append("w")
			 buf.append(i)
			 buf.append(":")
			 buf.append(weights(i))
			 buf.append(", ")
		}
		return buf + ")"
	}
}

class Layer(val inputNum: Int, val neurons : Array[Neuron]) {
	
	val outputs = new Array[Double](neurons.size)
	val errors = new Array[Double](neurons.size)
	var cumulatedError : Double = _
	
	def this(inputNum: Int, neuronNum: Int) = {
		
		//The last neuron inputs are used for the biais
		this(inputNum, Array.fill(neuronNum)(new Neuron(inputNum + 1)))
	}
	
	def run (inputs: Array[Double]) = {
		for(i <- 0 until outputs.size) {
			outputs(i) = neurons(i).run(inputs)
		}
	}
	
	override def toString() : String = {	
		var buf = new StringBuilder("[")
		neurons.foreach { neuron =>
			buf.append(neuron)
			buf.append(",")
		}
		buf.append("\n]\n")
		buf.toString
	}
}

object Perceptron {
	
	def loadXml(filePath : String) : Perceptron = {
	
		val perceptronXml = XML.load(filePath)
		val layers = new ListBuffer[Layer]
		
		for(layerXml <- perceptronXml \\ "layer") {
			val neurons = new ListBuffer[Neuron]
			val inputs : Int = (layerXml \ "@inputs").text.toInt
			
			for(neuronXml <- layerXml \\ "neuron") {
			
				val weights = new ListBuffer[Double]
				
				for(weightXml <- neuronXml \\ "weight") {
					weights += weightXml.text.toDouble
				}
				
				neurons += new Neuron(weights.toArray)
			}
			
			layers += new Layer(inputs, neurons.toArray)
		}
	
		new Perceptron(layers.toArray)
	}
}

class Perceptron (val layers : Array[Layer]){
	
	def this(perceptronInputNum : Int, neuronLayerNum: Array[Int]) = {
	
		this(new Array[Layer](neuronLayerNum.size))

		//Inputs for first layer equals the inputs of the perceptron
		var layerInputNum = perceptronInputNum

		//Creates the layers with each the right number of neurons
		for(i <- 0 until neuronLayerNum.size){

			//creates the layer
			layers(i) = new Layer(layerInputNum, neuronLayerNum(i))

			//the number of inputs of the next
			//layer is the current number of neurons
			layerInputNum = neuronLayerNum(i)
		}
	}
	
	def run (inputs: Array[Double]) : Array[Double] = {
		
		//The inputs for the first layer are the perceptron inputs
		var layerInput = inputs
		
		for(i <- 0 until layers.size) {
			
			//Calculates the outputs of the layer
			layers(i).run(layerInput)
			
			//The outputs are the inputs for the next layer
			layerInput = layers(i).outputs
		}
		
		//returns the outputs of the last layer
		return layerInput
	}
	
	def learn (inputs: Array[Double], outputs: Array[Double]) : Double = {
		
		val learnLevel : Double = 0.3
		val alfa : Double =  0.9
		
		//For all the layers but the first
		for(layerIndex <- layers.size - 1 to(1, -1)) {
			
			//current layer
			val layer = layers(layerIndex)
			//previous layer
			val previousLayer = layers(layerIndex - 1)
			
			//For every neuron in the current layer
			for(neuronIndex <- 0 until layer.neurons.size){
				
				//current neuron and error
				val neuron = layer.neurons(neuronIndex)
				val neuronError = layer.errors(neuronIndex)
				
				//For every neuron in the previous layer
				for(previusNeuronIndex <- 0 until previousLayer.neurons.size){

					neuron.pastErrors(previusNeuronIndex) = 
						learnLevel * neuronError * previousLayer.outputs(previusNeuronIndex) +
						alfa * neuron.pastErrors(previusNeuronIndex)
	
					neuron.weights(previusNeuronIndex) = neuron.pastErrors(previusNeuronIndex) + neuron.weights(previusNeuronIndex)
				}
				
				//updates for the biais
				neuron.pastErrors(previousLayer.neurons.size) =
					-learnLevel * neuronError + alfa * neuron.pastErrors(previousLayer.neurons.size)
				neuron.weights(previousLayer.neurons.size) = neuron.pastErrors(previousLayer.neurons.size) + neuron.weights(previousLayer.neurons.size)
			}
		}
		
		//For the first layer
		val layer = layers.head
		for(neuronIndex <- 0 until layer.neurons.size){
				
			//current neuron and error
			val neuron = layer.neurons(neuronIndex)
			val neuronError = layer.errors(neuronIndex)
				
				//For every input
				for(inputIndex <- 0 until inputs.size){
					
					neuron.pastErrors(inputIndex) = 
						learnLevel * neuronError * inputs(inputIndex) +
						alfa * neuron.pastErrors(inputIndex)
						
					neuron.weights(inputIndex) = neuron.pastErrors(inputIndex) + neuron.weights(inputIndex)
				}
				
				//updates for the biais
				neuron.pastErrors(inputs.size) =
					-learnLevel * neuronError + alfa * neuron.pastErrors(inputs.size)
				neuron.weights(inputs.size) = neuron.pastErrors(inputs.size) + neuron.weights(inputs.size)
			}
		
		//Calculates the cuadratic error between outputs and expected values
		//TODO: Use a functional style
		var cumul : Double = 0
		for (i <- 0 until outputs.size) {
			cumul += pow(outputs(i) - layers.last.outputs(i), 2)
		}
		sqrt(cumul)
	}
	
	def calculateErrors (inputs: Array[Double], outputs: Array[Double]) = {
		
		//For each layer from last to first
		for (layerIndex <- layers.size - 1 to(0,-1)) {
			//current layer
			val layer = layers(layerIndex)
			
			//For every neuron in the layer
			for(neuronIndex <- 0 until layer.neurons.size){
				
				//current neuron
				val neuron = layer.neurons(neuronIndex)
				//output for current neuron
				val neuronOutput = layer.outputs(neuronIndex)
										
				//Last factor of the error calculation
				if(layerIndex == layers.size - 1){ //Last layer
					layer.errors(neuronIndex) = neuronOutput * (1 - neuronOutput) * (outputs(neuronIndex) - neuronOutput)
				} else { //Hidden layers
					var tmp : Double = 0
					for(nextNeuronIndex <- 0 until layers(layerIndex + 1).neurons.size)
						tmp += layers(layerIndex + 1).errors(nextNeuronIndex) * layers(layerIndex + 1).neurons(nextNeuronIndex).weights(neuronIndex)
					
					layer.errors(neuronIndex) = neuronOutput * (1 - neuronOutput) * tmp
				}
				//Calculate the cumulated error of the layer
				//layer.cumulatedError =
				//	neuron.weights.reduceLeft( _ + layer.errors(neuronIndex) * _)
			}
			
        }
	}
	
	override def toString() : String = {
		var buf = new StringBuilder
		layers.foreach { layer => buf.append(layer.toString) }
		buf.toString
	}
	
	def toXml() : Elem = {
	
		val perceptronXml = 
		<perceptron layers={layers.size.toString}>
			{for (layer <- layers) yield
				<layer inputs={layer.inputNum.toString} neurons={layer.neurons.size.toString}>
					{for (neuron <- layer.neurons) yield
						<neuron inputs={neuron.weights.size.toString}>
							{for (weight <- neuron.weights) yield
								<weight>{weight.toString}</weight>
								}
						</neuron>
					}
				</layer>
			}
		</perceptron>
		
		return perceptronXml
	}
	
	def saveXml(filePath : String) = {
		XML.save(filePath, toXml(), "UTF-8", true, null)
	}
}

object NeuralNetworkApp extends Application {
	
	
	def generateXOR = {
	
		// We use a perceptron with 3 layers, the first with 5 neurons, the second with 10 and the las with only one
		val perceptron = new Perceptron(2,Array[Int](5, 10,1))

		val inputs = Array[Array[Double]](
			Array[Double](0,0),
			Array[Double](0,1),
			Array[Double](1,0),
			Array[Double](1,1)
		)
		val outputs = Array[Double](0, 1, 1, 0)
		
		println("Training neural network with XOR")
		//100 iterations to train the network
		for(i <- 0 until 100) {
			for(i <- 0 until inputs.size) {
				perceptron.run(inputs(i))
				perceptron.calculateErrors(inputs(i), Array[Double](outputs(i)))
				perceptron.learn(inputs(i), Array[Double](outputs(i)))
			}
		}
		
		println("Saving the perceptron configuration in XOR.xml")
		perceptron.saveXml("XOR.xml")
	}
	
	def runXOR = {
		
		println("Loading the perceptron configuration from XOR.xml")
		val perceptron = Perceptron.loadXml("XOR.xml")
		
		println("Evaluation of the inputs:")
		print("0,0 => ")
		perceptron.run(Array[Double](0, 0)).foreach(i =>{println(i)})
		
		print("0,1 => ")
		perceptron.run(Array[Double](0, 1)).foreach(i =>{println(i)})
		
		print("1,0 => ")
		perceptron.run(Array[Double](1, 0)).foreach(i =>{println(i)})
		
		print("1,1 => ")
		perceptron.run(Array[Double](1, 1)).foreach(i =>{println(i)})
	}
	
	override def main (args: Array[String]) {
		generateXOR
		runXOR
	}
}

