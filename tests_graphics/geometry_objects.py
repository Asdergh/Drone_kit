import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random as rd



plt.style.use("dark_background")
class File_treatmenter():
    def __init__(self, count_ot_elements, file_format=None, data_type=None,
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
            x_cores = np.random.normal(rd.randint(5, 5), 0.76, self.count_of_elems)
            y_cores = np.random.normal(rd.randint(5, 5), 0.76, self.count_of_elems)
            z_cores = np.random.normal(rd.randint(5, 5), 0.76, self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))
            print(self.basis_matrix)

        if self.data_type == "gamma":
            x_cores = np.random.gamma(rd.randint(0, 12), 12.76, self.count_of_elems)
            y_cores = np.random.gamma(rd.randint(0, 12), 12.76, self.count_of_elems)
            z_cores = np.random.gamma(rd.randint(0, 12), 12.76, self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))
            print(self.basis_matrix)

        if self.data_type == "chi_square":
            x_cores = np.random.chisquare(float(rd.randint(5, 6)), self.count_of_elems)
            y_cores = np.random.chisquare(float(rd.randint(5, 7)), self.count_of_elems)
            z_cores = np.random.chisquare(float(rd.randint(5, 8)), self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))
            
        if self.data_type == "stude_distrib":
            x_cores = np.random.standard_t(float(rd.randint(5, 45)), self.count_of_elems)
            y_cores = np.random.standard_t(float(rd.randint(5, 45)), self.count_of_elems)
            z_cores = np.random.standard_t(float(rd.randint(5, 45)), self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))

        if self.data_type == "weibul":
            x_cores = np.random.weibull(float(rd.randint(5, 45)), self.count_of_elems)
            y_cores = np.random.weibull(float(rd.randint(5, 45)), self.count_of_elems)
            z_cores = np.random.weibull(float(rd.randint(5, 45)), self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))
            
        if self.data_type == "fisher":
            x_cores = np.random.f(float(rd.randint(5, 45)), 12.76, self.count_of_elems)
            y_cores = np.random.f(float(rd.randint(5, 45)), 12.76, self.count_of_elems)
            z_cores = np.random.f(float(rd.randint(5, 45)), 12.76, self.count_of_elems)
            self.basis_matrix = np.dstack((x_cores, y_cores, z_cores))
        
        if self.bin_mode == False:
            with open(self.file_name, "w") as file:
                for (x, y, z) in self.basis_matrix[0]:
                    file.write(f"{x}\t{y}\t{z}\n")

        elif self.bin_mode == True:
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

    def __init__(self, x_cores, y_cores, z_cores=None,
                 cmap=None, alpha=None) -> None:
        self.x_cores = x_cores
        self.y_cores = y_cores
        self.x_grid, self.y_grid = np.meshgrid(self.x_cores, self.y_cores)
        self.z_cores = z_cores
        self.cmap = cmap
        self.alpha = alpha
        self.figure = plt.figure()
        self.axis_3d = self.figure.add_subplot(projection="3d")
    
    def create_core(self, operation_string: str):

        self.result_core_array = np.ones_like(self.x_grid)
        self.flag_dict = {
            "X": self.x_grid,
            "Y": self.y_grid
        }
        operation_list = ["+", "-", "*", "**", "/", "C", "S", "T"]
        for (item, str_object) in enumerate(operation_string):
            if operation_string[item] == "X":
                if operation_string[item - 1] not in operation_list:
                    self.result_core_array = self.flag_dict[str_object]
                elif operation_string[item - 1] == "+":
                    self.result_core_array += self.flag_dict[str_object]
                elif operation_string[item - 1] == "-":
                    self.result_core_array -= self.flag_dict[str_object]
                elif operation_string[item - 1] == "*":
                    self.result_core_array *= self.flag_dict[str_object]
                elif operation_string[item - 1] == "**":
                    self.result_core_array = self.result_core_array ** self.flag_dict[str_object]
                elif operation_string[item - 1] == "/":
                    self.result_core_array /= self.flag_dict[str_object]
                elif operation_string[item - 1] == "C":
                    self.result_core_array = np.cos(self.result_core_array + self.flag_dict[str_object])
                elif operation_string[item - 1] == "S":
                    self.result_core_array = np.sin(self.result_core_array + self.flag_dict[str_object])
                elif operation_string[item - 1] == "T":
                    self.result_core_array = np.tan(self.result_core_array + self.flag_dict[str_object])

            if operation_string[item] == "Y":
                if operation_string[item - 1] not in operation_list:
                    self.result_core_array = self.flag_dict[str_object]
                elif operation_string[item - 1] == "+":
                    self.result_core_array += self.flag_dict[str_object]
                elif operation_string[item - 1] == "-":
                    self.result_core_array -= self.flag_dict[str_object]
                elif operation_string[item - 1] == "*":
                    self.result_core_array *= self.flag_dict[str_object]
                elif operation_string[item - 1] == "**":
                    self.result_core_array = self.result_core_array ** self.flag_dict[str_object]
                elif operation_string[item - 1] == "/":
                    self.result_core_array /= self.flag_dict[str_object]
                elif operation_string[item - 1] == "C":
                    self.result_core_array = np.cos(self.result_core_array + self.flag_dict[str_object])
                elif operation_string[item - 1] == "S":
                    self.result_core_array = np.sin(self.result_core_array + self.flag_dict[str_object])
                elif operation_string[item - 1] == "T":
                    self.result_core_array = np.tan(self.result_core_array + self.flag_dict[str_object])
        
        return self.result_core_array

    def generate_graph(self, file_name=None, file_mode=False, distrib=None):
        
        if file_mode == True:
            if file_name == None:
                data = pd.read_csv(self.file_name, delimiter="\t")
                X_cores = np.array(data.iloc[:, 0])
                Y_cores = np.array(data.iloc[:, 1])
                Z_cores = np.array(data.iloc[:, 1])
            
            else:
                self.data = pd.read_csv(file_name, delimiter="\t")
                X_cores = np.array(data.iloc[:, 0])
                Y_cores = np.array(data.iloc[:, 1])
                Z_cores = np.array(data.iloc[:, 2])
            
            if distrib == "X":
                self.axis_3d.scatter(X_cores, Y_cores, Z_cores, c=X_cores, cmap=self.cmap, alpha=self.alpha, s=3.76)
            elif distrib == "Y":
                self.axis_3d.scatter(X_cores, Y_cores, Z_cores, c=Y_cores, cmap=self.cmap, alpha=self.alpha, s=3.76)
            elif distrib == "Z":
                self.axis_3d.scatter(X_cores, Y_cores, Z_cores, c=Z_cores, cmap=self.cmap, alpha=self.alpha, s=3.76)
            self.axis_3d.plot(X_cores, Y_cores, Z_cores, color="white", linewidth=0.26)
            plt.show()
        
        if file_mode == False:

            self.axis_3d.plot_surface(self.x_grid, self.y_grid, self.result_core_array,
                                      cmap=self.cmap, alpha=self.alpha)
            self.axis_3d.quiver(0, 0, 0, 3, 0, 0, color="blue")
            self.axis_3d.quiver(0, 0, 0, 0, 3, 0, color="red")
            self.axis_3d.quiver(0, 0, 0, 0, 0, 3, color="green")

            plt.show()
    
    def generate_animaiton(self, file_name=None, file_mode=None, distrib=None):

        def points(i):
            global x_cores, y_cores, z_cores
            self.axis_3d.scatter(x_cores, y_cores, z_cores, cmap=self.cmap, distrib=distrib, s=0.76)
        
        def surface(i):
            global x_cores, y_cores, z_cores
            self.axis_3d.plot_surface(x_cores, y_cores, z_cores, cmap=self.cmap, alpha=self.alpha)


        if file_mode == True:
            if file_name == None:
                data = pd.read_csv(self.fiel_name, delimiter="\t")
                x_cores = data[:, 0]
                y_cores = data[:, 1]
                z_cores = data[:, 2]

                animo = animation.FuncAnimation(self.figure, points, interval=100)
            
            else:
                data = pd.read_csv(file_name, delimiter="\t")
                x_cores = data[:, 0]
                y_cores = data[:, 1]
                z_cores = data[:, 2]

                animo = animation.FuncAnimation(self.figure, points, interval=100)
                plt.show()

        else:
            x_cores = self.x_grid
            y_cores = self.y_grid
            z_cores = self.result_core_array

            animo = animation.FuncAnimation(self.figure, surface, interval=100)
            plt.show()




x, y = np.linspace(-np.pi, np.pi, 100), np.linspace(-np.pi, np.pi, 100)
z_cores = np.linspace(-np.pi, np.pi, 100)
#x, y = np.meshgrid(x, y)
object = Geometry(x_cores=x, y_cores=y, cmap="twilight", alpha=0.76)
object_2 = File_treatmenter(file_name="distrib_data.txt", data_type="gamma", count_ot_elements=1000)
object_2.generate_data()
object.create_core("X *X +Y -CX")
#object.generate_graph(file_mode=False, file_name="distrib_data.txt", distrib="Z")
#
            
        



            




                


            
            

                    

        


        
        
        
                
        
                    
