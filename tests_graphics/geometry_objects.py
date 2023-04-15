import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animaiton
import random as rd


class File_treatmenter():
    def __init__(self, count_ot_elements, file_format, data_type: str,
                file_name=None, bin_mode=False) -> None:
        
        self.count_of_elems = count_ot_elements
        self.file_format = file_format
        self.data_type = data_type
        self.file_name = file_name
        self.bin_mode = bin_mode
        if self.bin_mode == True:
            self.file_name + ".bin"

    def generate_data(self):

        if self.data_type == "normal":
            x_cores = np.random.normal(rd.randint(0, 12), 0.76, 1000000)
            y_cores = np.random.normal(rd.randint(0, 12), 0.76, 1000000)
            z_cores = np.random.normal(rd.randint(0, 12), 0.76, 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))

        if self.data_type == "gamma":
            x_cores = np.random.gamma(rd.randint(0, 12), 0.76, 1000000)
            y_cores = np.random.gamma(rd.randint(0, 12), 0.76, 1000000)
            z_cores = np.random.gamma(rd.randint(0, 12), 0.76, 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))

        if self.data_type == "chi_square":
            x_cores = np.random.chisquare(float(rd.randint(0, 12)), 1000000)
            y_cores = np.random.chisquare(float(rd.randint(0, 12)), 1000000)
            z_cores = np.random.chsiquare(float(rd.randint(0, 12)), 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))
            
        if self.data_type == "stude_distrib":
            x_cores = np.random.standard_t(float(rd.randint(0, 12)), 1000000)
            y_cores = np.random.standard_t(float(rd.randint(0, 12)), 1000000)
            z_cores = np.random.standard_t(float(rd.randint(0, 12)), 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))

        if self.data_type == "weibul":
            x_cores = np.random.weibull(float(rd.randint(0, 12)), 1000000)
            y_cores = np.random.weibull(float(rd.randint(0, 12)), 1000000)
            z_cores = np.random.weibull(float(rd.randint(0, 12)), 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))
            
        if self.data_type == "fisher":
            x_cores = np.random.f(float(rd.randint(0, 12)), 0.76, 1000000)
            y_cores = np.random.f(float(rd.randint(0, 12)), 0.76, 1000000)
            z_cores = np.random.f(float(rd.randint(0, 12)), 0.76, 1000000)
            self.basis_matrix = np.stack((x_cores, y_cores, z_cores))
        
        if self.bin_mode == "False":
            with open(self.file_name, "w") as file:
                for (x, y, z) in self.basis_matrix:
                    file.write(f"{x}\t{y}\t{z}\n")

        elif self.bin_mode == "Bin":
            with open(self.file_name, "wb") as file:
                for (x, y, z) in self.basis_matrix:
                    file.write(f"{bin(x)}\t{bin(y)}\t{bin(z)}\n")
    
    def generate_ot_Array(self, ARRAY):
        if self.bin_mode == "False":
            with open(self.file_name, "w") as file:
                for (x, y, z) in ARRAY:
                    file.write(f"{x}\t{y}\t{z}\n")

        elif self.bin_mode == "Bin":
            with open(self.file_name, "wb") as file:
                for (x, y, z) in ARRAY:
                    file.write(f"{bin(x)}\t{bin(y)}\t{bin(z)}\n")
    


    

class Geometry(File_treatmenter):

    def __init__(self, x_cores, y_cores, z_cores) -> None:
        self.x_cores = x_cores
        self.y_cores = y_cores
        self.x_grid, self.y_grid = np.meshgrid(self.x_cores, self.y_cores)
        self.z_cores = z_cores
    
    def create_core(self, operation_string: str):

        self.result_core_array = []
        self.flag_dict = {
            "X": self.x_grid,
            "Y": self.y_grid,
            "Z": self.z_grid
        }
        curent_object = ""
        border_of_string = 0
        for (item, str_object) in enumerate(operation_string):
            if str_object == "X":
                curent_object = str_object
                self.result_core_array = self.flag_dict[curent_object]
                border_of_string = item
                break
            
            elif str_object == "Y":
                curent_object = str_object
                self.result_core_array = self.flag_dict[curent_object]
                break

            elif str_object == "Z":
                curent_object = str_object
                self.result_core_array = self.flag_dict[curent_object]
                break

                    

        


        
        
        
                
        
                    
