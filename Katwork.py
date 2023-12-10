import asyncio
from random import uniform
import math
from copy import deepcopy

class NeuralNetwork:

    def __init__(self, 
                 length_input_output_layout: tuple, 
                 neurons_layout: tuple=[], 
                 displacement_neuron: bool=False, 
                 function: str='liniar-function'
                 ) -> None:
        '''
            1. length_input_output_layout: 1 element asta e cifra care arata cati neuroni sunt in input layout. 2 element e cifra care arata cati neuroni sunt in ouput layout.
            2. neurons_layout: Cate elemente sunt in lista asta e cate coloane de neuroni sunt. Elementul este cifra care arata cate neuroni e in coloana.
            3. displacement_neuron: Neuronul de deplasare.
            4. function: dinumirea la functie care sa fie utilizata(sigmoid, hyperbolic-tangent, liniar-function).
        '''
        self.f = lambda x: (1 / (1 + math.e**(-x)) if function.lower() == 'sigmoid' else (math.e**(2*x) - 1) / (math.e**(2*x) + 1) if function.lower() == 'hyperbolic-tangent' else x if function.lower() == 'linear-function' else None)
        self.length_i_o = length_input_output_layout
        self.displacement_neuron = displacement_neuron
        self.neuralnetwork = []

        if self.f == None: return 'error'

        #Generate hidden layouts
        for l_i, l in enumerate(neurons_layout):
            layout = [] 
            for n in range(l):
                neuron = []
                for s in range((neurons_layout[l_i-1] if l_i > 0 else self.length_i_o[0]) + (1 if self.displacement_neuron else 0)): neuron.append(uniform(-0.5, 1))

                layout.append(neuron)

            self.neuralnetwork.append(layout)

        #Generate output layout
        layout = []
        for n in range(self.length_i_o[1]):
            neuron = []
            for w in range((len(self.neuralnetwork[-1]) if len(self.neuralnetwork) > 0 else self.length_i_o[0]) + (1 if self.displacement_neuron else 0)): neuron.append(uniform(-0.5, 1))

            layout.append(neuron)
        
        self.neuralnetwork.append(layout)

    async def MSE(self, 
            all_difference: int|float, 
            example_range: int
            ) -> float:
        '''
            1. all_difference: calculul la toate exemplele: (a_1 - i_1)^2 + (a_2 - i_2)^2 ... (a_n - i_n)^2
            2. example_length: numarul de exemple.
        '''
        return all_difference / example_range
    
    async def root_MSE(self, 
                 all_difference: int|float, 
                 example_range: int
                 ) -> float:
        '''
            1. all_difference: calculul la toate exemplele: (a_1 - i_1)^2 + (a_2 - i_2)^2 ... (a_n - i_n)^2
            2. example_length: numarul de exemple.
        '''
        return math.sqrt(all_difference / example_range)
    
    async def update_w(self
                 ) -> None:
        for l_i, l in enumerate(self.neuralnetwork):
            for n in range(len(l)):
                for w in range(len(l[n])): self.neuralnetwork[l_i][n][w] = uniform(-0.5, 1)
    
    async def backpropagation(self, 
                        example_list: tuple|list, 
                        maximal_era: int, 
                        learn_speed: float|int=0.7, 
                        moment: int|float=0.3
                        ) -> list:
        '''
            1. example: lista cu exemple dupa care NN se invata. Fiecare element din lista asta e un exemplu, un exemplu consta din lista care are in ea atatia neuroni de intrare cat ati indicat la crearea NN si dupa urmeaza neuronii de iesire, tot cat ati indicat la crearea la NN(Rezultatul croect).
            2. maximal_era: era maximala pana la care poate sa se invete.
            3. learn_speed: Rata cu care se algoritmul de invatare ajusteaza parametrii sai in functie de datele de antrenament. O valoare mai a vitedezei de invatare poate duce la convergenta mai rapida, dar poate provoca si oscilatii sau divergenta.
            4. moment: Rata care accelereaza convergenta si care eviti minimul local.
            5. show_data: arata datele despre evolutie, ce secol sau mileniu.
        '''

        back_neuralnetwork = deepcopy(self.neuralnetwork)
        back_neuralnetwork.reverse()
        back_neuralnetwork.append([[] for n in range(self.length_i_o[0])])

        gradient = deepcopy(back_neuralnetwork)
        w_shift = deepcopy(back_neuralnetwork)

        delta1 = lambda output_ideal, output_actual: (output_ideal - output_actual) * ((1 - output_actual) * output_actual) #Formula la delta pentru output
        delta2 = lambda neuron, sum_wd: ((1 - neuron) * neuron) * sum_wd #Formula la delta pentru hidden
        grad = lambda neuron, delta: delta * neuron
        shift = lambda E, grad, A, last_shift: E*grad + A*last_shift

        for era in range(maximal_era):
            error = [0 for i in range(self.length_i_o[1])]

            for iteration, example in enumerate(example_list):
                output = await self.run(input_layout=example[:self.length_i_o[0]], get_layout=True)

                actual_output = deepcopy(output[0])
                all_data = deepcopy(output[1])
                all_data.reverse()
                
                #Output operation
                delta_output_layout = [0 for d in range(self.length_i_o[1])]
                for i, n in enumerate(actual_output):
                    error[i] += (example[self.length_i_o[0]:][i] - n)**2
                    delta_output_layout[i] = delta1(output_ideal=example[self.length_i_o[0]:][i], output_actual=n)
                
                #Hidden and Input operation
                copy_nn = deepcopy(back_neuralnetwork)
                for i, l in enumerate(copy_nn):
                    if i == len(copy_nn)-1: break

                    delta_hidden_layout = [0 for d in range(len(all_data[i+1]))] 
                    for i2, n in enumerate(copy_nn[i+1]):
                        sum_wd = 0
                        for i3, n2 in enumerate(l):
                            sum_wd += n2[i2] * delta_output_layout[i3]
                            
                            gradient[i][i3][i2] = grad(all_data[i+1][i2], delta_output_layout[i3])
                            w_shift[i][i3][i2] = shift(learn_speed, gradient[i][i3][i2], moment, 0 if iteration == 0 else w_shift[i][i3][i2])
                            back_neuralnetwork[i][i3][i2] += w_shift[i][i3][i2]
                        
                        delta_hidden_layout[i2] = delta2(all_data[i+1][i2], sum_wd)

                    delta_output_layout = delta_hidden_layout

                neuralnetwork = deepcopy(back_neuralnetwork)
                neuralnetwork.pop(-1)
                neuralnetwork.reverse()
                self.neuralnetwork = neuralnetwork
                        
            for i in range(self.length_i_o[1]):
                error[i] = await self.MSE(all_difference=error[i], example_range=len(example_list))

            print(f'---------\nEra: {era}\nError: {error}')
        
        return error

    async def run(self, 
            input_layout: list, 
            get_layout: bool=False
            ) -> tuple[list[float]|None]:
        '''
            1. input_layout: lista cu parametrele de intrare(cati neuroni de intrare sunt atitea parametre trebuie sa fie, un element arata starea la un neuron de intrare).
        '''

        input_data = input_layout + ([1] if self.displacement_neuron else [])
        all_data = [input_data]
        for l in self.neuralnetwork:
            layout_output_data = []
            for n in l:
                result = 0
                for i, w in zip(input_data, n):
                    result += self.f(i) * w

                layout_output_data.append(self.f(result) if result != 0 else result)
            input_data = layout_output_data + ([1] if self.displacement_neuron else [])
            all_data.append(input_data)

        return input_data[:-1] if self.displacement_neuron else input_data, all_data if get_layout else None