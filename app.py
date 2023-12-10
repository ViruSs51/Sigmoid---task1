import asyncio

import data_operation
import Katwork

class App(data_operation.File):

    def __init__(self, 
                 config_file: str='config.json', 
                 get_weighting: bool=False
                 ) -> None:
        '''
            1. config_file: drumul spre fisierul config.json
            2. get_weighting: False/True zice daca sa ia sau nu cantarurile salvate in config.json pentru reteaua neuronala.
        '''
        self.config_path = config_file
        self.config = self.get_json(file=self.config_path)
        self.nn_weighting = self.config['neuralnetwork']['weighting']

        self.sigmoid = Katwork.NeuralNetwork(length_input_output_layout=(8, 1), 
                                             neurons_layout=[10, 10, 10], 
                                             displacement_neuron=False, 
                                             function='sigmoid') #Genereaza reteaua neuronala
        
        self.get_weighting = get_weighting

    async def run(self,
                  type: str='learn'
            ) -> None:
        '''
            type: 'learn' sau etc. zice daca sa invete reteaua neuronala ori nu.
        '''
        if self.get_weighting:
            returned = await self.load_weighting()
            if returned is not None:
                print(returned)

        if type == 'learn':
            task = asyncio.create_task(self.learn())

            await task
        else:
            task = asyncio.create_task(self.input(input_data=[float(e) for e in type.split(',')], binar=True))

            print((await task)[0])

    async def load_weighting(self
                       ) -> str|None:
        if self.nn_weighting != None: self.sigmoid.neuralnetwork = self.nn_weighting
        else: return -1

    async def input(self, 
              input_data: list,
              binar: bool=False
              ) -> tuple:
        '''
            input_data: datele de intrare pentru reteaua neuronala.
            binar: False/True zice sa returneze output-ul retelei neuronale in forma de procente ori ca 1/0
        '''
        f = lambda x: 1 if x >= 0.5 else 0
        output_data = await self.sigmoid.run(input_layout=input_data)

        return output_data if not binar else ([f(o) for o in output_data[0]], output_data[1])

    async def learn(self
              ) -> None:
        data = tuple([[int(e) if i >= 8 or e == '0' else float(e)
                       for i, e in enumerate(example[1:])] 
                      for example in self.get_csv(file=self.config['path']['nn_data_learn'])])
        
        error = await self.sigmoid.backpropagation(example_list=data, maximal_era=100, learn_speed=0.7, moment=0.3)

        self.config['neuralnetwork']['error'] = error
        self.config['neuralnetwork']['weighting'] = self.sigmoid.neuralnetwork

        self.put_json(f'{self.config_path}', self.config)