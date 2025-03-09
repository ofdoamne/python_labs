import os
import math
import copy

class Matrix:
    def __init__(self, matrix):
        self._matrix = matrix
    
    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    def add(self, second_matrix):
        """
        метод сложения двух матриц

        Аргументы:
        second_matrix (list): Вторая матрица, с которой будет выполнено сложение. Должна быть такого же размера, как и текущая матрица

        Возвращаемое значение:
        list: Новая матрица, являющаяся результатом сложения текущей матрицы и второй матрицы

        Исключения:
        ValueError: Если матрицы имеют разные размеры (не одинаковое количество строк и столбцов)
        """
        try:
            if len(self._matrix) != len(second_matrix._matrix) or len(self._matrix[0]) != len(second_matrix._matrix[0]):
                raise ValueError("Матрицы должны быть одинакового размера для сложения")
            result = []
            # получаем соответсвующие друг другу строки исходных матриц
            for row_matrix, row_second_matrix in zip(self._matrix, second_matrix._matrix):
                # получаем соответсвующие друг другу элменты из строк, складываем их и записываем в строку результата
                row_result = [a + b for a, b in zip(row_matrix, row_second_matrix)]
                # формируем матрицу с результатом сложения
                result.append(row_result)
            return result
        except ValueError as e:
            print(f"Ошибка: {e}. Попробуйте снова.")
    
    def subtract(self, second_matrix):
        """
        метод вычитания двух матриц

        Аргументы:
        second_matrix (list): Вторая матрица, из которой будет вычтена текущая матрица. Должна быть такого же размера, как и текущая матрица

        Возвращаемое значение:
        list: Новая матрица, являющаяся результатом вычитания второй матрицы из текущей матрицы

        Исключения:
        ValueError: Если матрицы имеют разные размеры (не одинаковое количество строк и столбцов)
        """
        if len(self._matrix) != len(second_matrix._matrix) or len(self._matrix[0]) != len(second_matrix._matrix[0]):
            raise ValueError("Матрицы должны быть одинакового размера для вычитания")
        result = []
        for row_matrix, row_second_matrix in zip(self._matrix, second_matrix._matrix):
            row_result = [a - b for a, b in zip(row_matrix, row_second_matrix)]
            result.append(row_result)
        return result

    def multiply_by_vector(self, vector):
        """
        метод умножения матрицы на вектор

        Аргументы:
        vector (list): Вектор (список), на который будет умножена матрица. Его длина должна совпадать с количеством столбцов матрицы

        Возвращаемое значение:
        list: Вектор (список), содержащий результаты умножения матрицы на вектор

        Исключения:
        ValueError: Если количество столбцов матрицы не совпадает с количеством элементов вектора
        """
        result = []
        # получаем строку матрицы
        for row_matrix in self._matrix:
            # получаем элемент строки матрицы и соответсвующий элемент вектора
            row_result_multiply = [a * b for a, b in zip(row_matrix, vector)]
            # результат умножения во всей строке
            row_sum_result = sum(row_result_multiply)
            # запись в список
            result.append(row_sum_result)
        return result

    def multiply_by_matrix(self, second_matrix):
        """
        метод умножения матриц

        Аргументы:
        second_matrix (list): Вторая матрица (список списков), на которую будет умножена текущая матрица
                      Количество столбцов первой матрицы должно совпадать с количеством строк второй

        Возвращаемое значение:
        list: Матрица (список списков), являющаяся результатом умножения текущей матрицы на переданную

        Исключения:
        ValueError: Если количество столбцов первой матрицы не совпадает с количеством строк второй матрицы
        """
        if len(self._matrix[0]) != len(second_matrix._matrix):
            raise ValueError("Число столбцов A должно совпадать с числом строк B")
        result = [[0] * len(second_matrix._matrix[0]) for _ in range(len(self._matrix))]
        for i in range(len(self._matrix)):  # Для каждой строки A
            for j in range(len(second_matrix._matrix[0])):  # Для каждого столбца B
                for k in range(len(second_matrix._matrix)):  # Скалярное произведение
                    result[i][j] += self._matrix[i][k] * second_matrix._matrix[k][j]
        return result

    def transpose(self):
        """
        метод транспонирования матрицы

        преобразует строки матрицы в столбцы и наоборот.

        Возвращаемое значение:
        list: Матрица (список списков), являющаяся транспонированной версией текущей матрицы.
        """
        return [list(row) for row in zip(*self._matrix)]

    def determinant(self):
        """
        метод нахождения определителя матрицы

        Поддерживаются матрицы размером 2x2 и 3x3

        Возвращаемое значение:
        float: Определитель матрицы

        Исключения:
        ValueError: Если матрица не является квадратной или имеет размер, отличный от 2x2 или 3x3
        """
        if len(self._matrix) == 2:
            return self._matrix[0][0] * self._matrix[1][1] - self._matrix[0][1] * self._matrix[1][0]
        
        elif len(self._matrix) == 3:
            a, b, c = self._matrix[0]
            d, e, f = self._matrix[1]
            g, h, i = self._matrix[2]

            return (a * (e * i - f * h) - b * (d * i - f *g) + c * (d * h - e * g))
        else:
            raise ValueError("Поддерживаются только матрицы размером 2x2 и 3x3")

    def solve_gaussian(self, vector):  # vector — это правая часть уравнений
        n = len(self._matrix)
        matrix_copy = copy.deepcopy(self._matrix)  # Создаём копию исходной матрицы

        # Прямой ход (приведение к верхнетреугольному виду)
        for i in range(n):
            # Ищем максимальный элемент в столбце для стабильности
            max_row = max(range(i, n), key=lambda r: abs(matrix_copy[r][i]))
            
            if i != max_row:
                # Меняем местами строки в копии матрицы
                matrix_copy[i], matrix_copy[max_row] = matrix_copy[max_row], matrix_copy[i]
                vector[i], vector[max_row] = vector[max_row], vector[i]

            # Проверка на ноль в ведущем элементе
            if matrix_copy[i][i] == 0:
                print(f"Невозможно решить систему, так как элемент на диагонали {i}-й строки равен 0")
                return None  # Если элемент на диагонали равен нулю, решение невозможно

            # Делим строку на ведущий элемент
            for j in range(i + 1, n):
                if matrix_copy[j][i] != 0:
                    factor = matrix_copy[j][i] / matrix_copy[i][i]
                    vector[j] -= factor * vector[i]
                    for k in range(i, n):
                        matrix_copy[j][k] -= factor * matrix_copy[i][k]

        # Обратный ход (решаем систему)
        x = [0] * n
        for i in range(n - 1, -1, -1):
            sum_ = 0
            for j in range(i + 1, n):
                sum_ += matrix_copy[i][j] * x[j]
            
            try:
                if matrix_copy[i][i] == 0:
                    raise ZeroDivisionError(f"Невозможно решить систему, так как элемент на диагонали {i}-й строки равен 0")
                
                x[i] = round((vector[i] - sum_) / matrix_copy[i][i])
            
            except ZeroDivisionError as e:
                print(e)  # Выводим сообщение о том, что невозможно продолжить решение
                return None

        return x

class Vector:
    def __init__(self, vector_coords):
        """
        Конструктор для создания объекта вектора

        Параметры:
            vector_coords (list): Список из двух чисел, представляющий координаты вектора

        Инициализирует объект вектора с переданными координатами
        """
        self._vector_coords = vector_coords
    
    @property
    def vector(self):
        """
        Свойство для получения координат вектора

        Возвращает:
            list: Список, содержащий два числа, представляющих координаты вектора
        """
        return [int(coord) if coord.is_integer() else coord for coord in self._vector_coords]
    
    @vector.setter
    def vector(self, values):
        """
        Метод задает новое значение координат вектора
    
        Параметры:
            values (list): Список, содержащий два числа, которые будут присвоены вектору
            
        Пример:
            vector = Vector([1, 2])
            vector.vector = [3, 4]  # Координаты вектора изменятся на [3, 4]
        
        Исключения:
            ValueError: Если переданный список не содержит ровно два элемента
        """
        self._vector_coords = values

    def vector_length(self):
        """
        Метод вычисляет длину (модуль) вектора

        Возвращает:
            float: Длина (модуль) вектора

        Пример:
            vector = Vector([3, 4])
            print(vector.vector_length())  # Ожидаем: 5.0
        """
        return round(math.sqrt(sum(comp**2 for comp in self._vector_coords)))
    
    def dot_product_of_two_vectors(self, second_vector):
        """
        Метод вычисляет скалярное произведение двух векторов
        Аргументы:
            second_vector (Vector): Второй вектор для вычисления скалярного произведения

        Возвращает:
            float: Результат скалярного произведения двух векторов

        Пример:
            vector1 = Vector([1, 2])
            vector2 = Vector([3, 4])
            result = vector1.dot_product_of_two_vectors(vector2)  # Ожидаемый результат: 11 (1*3 + 2*4)
        """
        return sum(a * b for a, b in zip(self._vector_coords, second_vector._vector_coords))

    def angle_between_two_vectors(self, second_vector):
        """
        Вычисляет угол между двумя векторами в градусах

        Условия:
            - Если один из векторов является нуль-вектором, вызывается ошибка ValueError
            - Если угол меньше 1 градуса, результат округляется до "0°"
            - Угол возвращается в формате строки с символом °

        Формула:
            cos(θ) = (A · B) / (|A| * |B|)
            θ = acos(cos(θ))

        Параметры:
            second_vector (Vector): Второй вектор для вычисления угла

        Возвращает:
            str: Угол между векторами в градусах, округленный до целого числа с символом °

        Исключения:
            ValueError: Если один из векторов является нуль-вектором

        Пример:
            vector1 = Vector([1, 0])
            vector2 = Vector([0, 1])
            print(vector1.angle_between_two_vectors(vector2))  # Ожидаем: "90°"
        """
        try:
            # Проверка на нуль-вектора
            if self.vector_length() == 0 or second_vector.vector_length() == 0:
                raise ValueError("Невозможно вычислить угол с нуль-вектором.")
            
            cos_angle = self.dot_product_of_two_vectors(second_vector) / (self.vector_length() * second_vector.vector_length())
            
            # Ограничение значения косинуса для предотвращения ошибок округления -1 и 1
            cos_angle = max(-1, min(1, cos_angle))
            
            # вычисление угла
            angle = math.degrees(math.acos(cos_angle))

            # Порог в 1 градус для "нулевого угла"
            if abs(angle) < 1:
                return "0°"
            
            # округляем и выводим результат со знаком °
            return f"{round(angle)}°"
        
        except ValueError as e:
            return str(e)
    
    def __add__(self, second_vector):
        """
        Метод выполняет поэлементное сложение текущего вектора с другим вектором

        Формула:
            Результат = [x1 + x2, y1 + y2, ...]

        Параметры:
            second_vector (Vector): Второй вектор, который нужно сложить с текущим

        Возвращает:
            list: Список, содержащий координаты нового вектора, полученного в результате сложения

        Пример:
            vector1 = Vector([1, 2])
            vector2 = Vector([3, 4])
            print(vector1 + vector2)  # Ожидаем: [4, 6]
        """
        return [a + b for a, b in zip(self._vector_coords, second_vector._vector_coords)]

    def __sub__(self, second_vector):
        """
        Метод выполняет поэлементное вычитание другого вектора из текущего

        Формула:
            Результат = [x1 - x2, y1 - y2, ...]

        Параметры:
            second_vector (Vector): Второй вектор, который нужно вычесть из текущего

        Возвращает:
            list: Список, содержащий координаты нового вектора, полученного в результате вычитания

        Пример:
            vector1 = Vector([5, 7])
            vector2 = Vector([2, 3])
            print(vector1 - vector2)  # Ожидаем: [3, 4]
        """
        return [a - b for a, b in zip(self._vector_coords, second_vector._vector_coords)]

    def scalar_multiplication_of_vector(self, scalar):
        """
        Метод выполняет умножение вектора на скаляр

        Формула:
            Результат = [x * scalar, y * scalar, ...]

        Параметры:
            scalar (int или float): Число, на которое нужно умножить вектор

        Возвращает:
            list: Список, содержащий координаты нового вектора, полученного в результате умножения на скаляр
            В случае ошибки возвращается строка с описанием ошибки

        Исключения:
            ValueError: Если переданный параметр scalar не является числом (int или float)

        Пример:
            vector = Vector([2, 4])
            print(vector.scalar_multiplication_of_vector(3))  # Ожидаем: [6, 12]

            vector = Vector([0, 1])
            print(vector.scalar_multiplication_of_vector("a"))  # Ожидаем: "Скаляр должен быть числом."
        """
        try:
            if isinstance(scalar, (int, float)):
                return [round(comp * scalar) for comp in self._vector_coords]
            else:
                raise ValueError("Скаляр должен быть числом.")
        except ValueError as e:
            return str(e)

    def check_vector_collinearity(self, second_vector):
        """
        Метод проверяет коллинеарность двух векторов

        Коллинеарность означает, что один вектор можно представить как масштабированную версию другого
        Метод учитывает случаи, когда компоненты обоих векторов равны нулю

        Параметры:
            second_vector (Vector): Второй вектор для проверки коллинеарности

        Возвращает:
            str: "векторы коллинеарны", если векторы коллинеарны
                "векторы не коллинеарны", если векторы не коллинеарны

        Алгоритм:
            - Если одна из компонент равна нулю, а другая нет, векторы не коллинеарны
            - Если обе компоненты равны нулю, это не влияет на проверку коллинеарности
            - Если ни одна из компонент не равна нулю, проверяется их отношение (коэффициент пропорции)
            - Если отношение меняется, векторы не коллинеарны

        Пример:
            vector1 = Vector([2, 4])
            vector2 = Vector([1, 2])
            print(vector1.check_vector_collinearity(vector2))  # Ожидаем: "векторы коллинеарны"

            vector3 = Vector([1, 0])
            vector4 = Vector([0, 1])
            print(vector3.check_vector_collinearity(vector4))  # Ожидаем: "векторы не коллинеарны"
        """
        ratio = None
        for a, b in zip(self._vector_coords, second_vector._vector_coords):
            # Проверка на несовпадение нулевых компонентов
            if (a == 0 and b != 0) or (a != 0 and b == 0):
                return "не коллинеарны"
            
            if a == 0 and b == 0:
                continue

            if a != 0:  # Если a != 0, вычисляем текущий коэффициент пропорции
                current_ratio = b / a
                if ratio is None:
                    ratio = current_ratio
                elif current_ratio != ratio:
                    return "не коллинеарны"
        return "коллинеарны" 

class EquationSolver:
    def __init__(self, coefficients, parameters=None):
        """
        Инициализация класса.

        :param coefficients: Словарь коэффициентов уравнения (например, {'a': 1, 'b': -5, 'c': 2}).
        :param parameters: Словарь дополнительных параметров уравнения (например, {'condition': 'real roots only'}).
        """
        self._coefficients = coefficients
        self._parameters = parameters if parameters else {}

    @property
    def coefficients(self):
        return self._coefficients
    
    @coefficients.setter
    def coefficients(self, new_coefficients):
        self._coefficients = new_coefficients

    @property
    def parameters(self):
        return self._parameters

    @parameters.setter
    def parameters(self, new_parameters):
        self._parameters = new_parameters
    
    def solve_linear(self):
        a = self._coefficients.get("a", 0)
        b = self._coefficients.get("b", 0)

        if a == 0:
            if b == 0:
                return f"Бесконечное множество решений"
            else:
                return f"Решений нет"
        
        return f"Решение: х = {round((-b / a), 2)}"
    
    def solve_quadratic(self):
        a = self._coefficients.get("a", 0)
        b = self._coefficients.get("b", 0)
        c = self._coefficients.get("c", 0)

        # Вычисляем дискриминант один раз
        discriminant = b**2 - 4*a*c

        # Проверка параметра 'condition', если он задан
        condition = self._parameters.get("condition", "").lower()

        if a == 0:
            return self.solve_linear()

        if condition == "real roots only" and discriminant < 0:
            return "Действительных корней нет"

        if condition == "positive roots only" and discriminant >= 0:
            x1 = (-b + discriminant**0.5) / (2 * a)
            x2 = (-b - discriminant**0.5) / (2 * a)
            if x1 <= 0 and x2 <= 0:
                return "Положительных корней нет"

        if discriminant < 0:
            return "Действительных корней нет"

        elif discriminant == 0:
            x = round(-b / (2 * a), 2)
            return f"Решение: х = {x}"

        else:
            x1 = round((-b + discriminant**0.5) / (2 * a), 2)
            x2 = round((-b - discriminant**0.5) / (2 * a), 2)
            
            # Убедимся, что возвращаем корни в правильном порядке
            return f"Решение: х1 = {x1}, х2 = {x2}"

class FunctionAnalyzer:
    def __init__(self, func, interval):
        """
        Инициализация класса.

        :param func: Функция, для которой нужно найти min/max (например, lambda x: x**2 - 4*x + 4).
        :param interval: Кортеж (a, b), задающий интервал для поиска min/max.
        """
        self._func = func
        self._interval = interval

    def evaluate_at(self, x):
        """Вычисление значения функции в точке x."""
        return self._func(x)

    def find_min_max(self, num_points=1000):
        """
        Поиск минимального и максимального значений функции на заданном интервале.

        :param num_points: Количество точек, на которые делится интервал для численного поиска.
        :return: Кортеж (min_value, max_value).
        """
        # Вычисляем шаг
        step = (self._interval[1] - self._interval[0]) / num_points
        x_values = [self._interval[0] + i * step for i in range(num_points + 1)]
        
        # Массив значений функции для всех точек
        y_values = [self.evaluate_at(x) for x in x_values]
        
        # Нахождение минимального и максимального значений
        min_value = min(y_values)
        max_value = max(y_values)
        
        return min_value, max_value

def clear_screen():
    """Очищает консоль в зависимости от операционной системы"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main_menu():
    clear_screen()
    while True:
        clear_screen()
        choice =  input("Выбор действия: \n 1. Работа с векторами \n 2. Работа с матрицами \n 3. Решение уравнений \nВыберите действие: ")
        if choice == "1":
            vector_menu()
        elif choice == "2":
            matrix_menu()
        elif choice == "3":
            equation_solver_menu()
        else:
            clear_screen()
            print("Некорректный выбор. Попробуйте снова")
            
def vector_menu():

    def validate_vector_input(vector_name):
        while True:
            try:
                input_str = input(f"Введите координаты {vector_name} вектора (два числа через пробел): ")
                list_coefficients = input_str.split()  # Разделяем строку на элементы
                
                # Проверка, что все элементы могут быть приведены к типу float
                try:
                    list_coefficients = [float(el) for el in list_coefficients]
                except ValueError:
                    raise ValueError("Ошибка: все элементы должны быть числами.")

                # Проверка, что введены два числа
                if len(list_coefficients) != 2:
                    raise ValueError("Ошибка: координаты должны быть два числа.")

                return list_coefficients  # Возвращаем корректный ввод

            except ValueError as e:
                clear_screen()
                print(e)  # Выводим сообщение об ошибке, если она произошла


    clear_screen()   
    first_vector = Vector(validate_vector_input("первого"))
    second_vector = Vector(validate_vector_input("второго"))
    clear_screen()

    while True:
        clear_screen()
        operation = input("Выбор операции:\n"
                  "1. Длина вектора.\n"
                  "2. Скалярное произведение двух векторов.\n"
                  "3. Угол между двумя векторами.\n"
                  "4. Сложение и вычитание векторов.\n"
                  "5. Умножение вектора на скаляр.\n"
                  "6. Проверка коллинеарности векторов.\n"
                  "7. Вернуться назад.\n"
                  "Выберите операцию: ")

        clear_screen()
            
        if operation == "1":
            while True:
                choice_vector = input(f"Выбор вектора для вычисления длины: \n 1. {first_vector.vector} \n 2. {second_vector.vector} \nВыберите вектор: ")
                if choice_vector == "1":
                    clear_screen()
                    print(f"Результат: {first_vector.vector_length()}")
                    input("Нажмите Enter для продолжения...")
                    break  
                elif choice_vector == "2":
                    clear_screen()
                    print(f"Результат: {second_vector.vector_length()}")
                    input("Нажмите Enter для продолжения...")
                    break  
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "2":
            clear_screen()
            print(f"Скалярное произведение векторов: {first_vector.vector} и {second_vector.vector} равно {first_vector.dot_product_of_two_vectors(second_vector)}")
            input("Нажмите Enter для продолжения...")
            clear_screen()

        elif operation == "3":
            clear_screen()
            print(f"Угол между векторами: {first_vector.vector} и {second_vector.vector} равен {first_vector.angle_between_two_vectors(second_vector)}")
            input("Нажмите Enter для продолжения...")
            clear_screen()

        elif operation == "4":
            while True:
                choice_operation = input("Выберите, какое действие выполнить: \n 1. Сложение \n 2. Вычитание \n Выберите действие: ")
                clear_screen()
                if choice_operation == "1":
                    print(f"Сумма векторов: {first_vector.vector} и {second_vector.vector} равна {first_vector + second_vector}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break
                elif choice_operation == "2":
                    choice_vector = input(f"Выберите, какое действие выполнить:\n"
                      f"1. Вычитание из вектора {first_vector.vector} вектор {second_vector.vector}\n"
                      f"2. Вычитание из вектора {second_vector.vector} вектор {first_vector.vector}\n"
                      f"Выберите действие: ")
                    clear_screen()
                    while True:
                        if choice_vector == "1":
                            print(f"Результат вычисления: {first_vector - second_vector}")
                            input("Нажмите Enter для продолжения...")
                            clear_screen()
                            break
                        elif choice_vector == "2":
                            print(f"Результат вычисления: {second_vector - first_vector}")
                            input("Нажмите Enter для продолжения...")
                            clear_screen()
                            break
                        else:
                            clear_screen()
                            print("Некорректный выбор. Попробуйте снова")
                    break
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "5":

            while True:
                try:
                    scalar = float(input("Введите значение для скаляра: "))
                    clear_screen()  # Очищаем экран после успешного ввода
                    break  # Выход из этого цикла, если все прошло успешно
                except ValueError:
                    clear_screen()
                    print("Ошибка: введено не число. Попробуйте снова.")
                    continue  # Повторно запрашиваем ввод только скаляра

            while True:    
                choice_vector = input(f"Выберите вектор для умножения: \n 1. {first_vector.vector} \n 2. {second_vector.vector} \nВыберите вектор: ")
                clear_screen()
            
                if choice_vector == "1":
                    print(f"Результат вычисления: {first_vector.scalar_multiplication_of_vector(scalar)}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break

                elif choice_vector == "2":
                    print(f"Результат вычисления: {second_vector.scalar_multiplication_of_vector(scalar)}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break
                
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")
      
        elif operation == "6":
            print(f"Векторы: {first_vector.vector} и {second_vector.vector} {first_vector.check_vector_collinearity(second_vector)}")
            input("Нажмите Enter для продолжения...")
            clear_screen()

        elif operation == "7":
            break

        else:
            clear_screen()
            print("Некорректный выбор. Попробуйте снова")

def matrix_menu():

    def validate_matrix_input(matrix_name, rows, cols):
        while True:
            try:
                print(f"Введите элементы {matrix_name} матрицы (каждая строка через пробел):")
                matrix = []
                for i in range(rows):
                    row_str = input(f"Строка {i + 1}: ")
                    # Преобразуем строку в список чисел и обрабатываем ошибку, если ввод не является числом
                    row = row_str.split()
                    for element in row:
                        if not element.isdigit():  # Проверяем, является ли элемент числом
                            raise ValueError(f"'{element}' не является числом...")
                    row = list(map(int, row))  # Преобразуем строку в целые числа

                    if len(row) != cols:  # Проверяем, что строка имеет правильное количество элементов
                        raise ValueError(f"Каждая строка {matrix_name} матрицы должна содержать {cols} элемента(ов).")
                    matrix.append(row)
                
                return matrix  # Возвращаем матрицу как список списков

            except ValueError as e:
                clear_screen()
                print(f"Ошибка: {e} Попробуйте снова.")

    def validate_matrix_size(matrix_name):
        while True:
            try:
                matrix_size_str = input(f"Введите размерность {matrix_name} матрицы (количество строк и столбцов через пробел): ").strip()
            
                # Проверяем, что строка содержит только цифры и пробелы
                if not all(c.isdigit() or c.isspace() for c in matrix_size_str):
                    raise ValueError("Ошибка: введите два целых числа через пробел.")
                
                matrix_size = list(map(int, matrix_size_str.split()))  # Преобразуем ввод в список чисел
                
                if len(matrix_size) != 2:
                    raise ValueError("Ошибка: необходимо ввести два числа через пробел.")
                
                if matrix_size == [1, 1]:
                    raise ValueError("Ошибка: программа не поддерживает матрицы размерностью 1x1.")
                
                return matrix_size  # Возвращаем список из двух чисел
            
            except ValueError as e:
                clear_screen()
                print(e)
    
    clear_screen()
    first_matrix_size = validate_matrix_size("первой")
    second_matrix_size = validate_matrix_size("второй")
    clear_screen()
    first_matrix = Matrix(validate_matrix_input("первой", first_matrix_size[0], first_matrix_size[1]))
    second_matrix = Matrix(validate_matrix_input("второй", second_matrix_size[0], second_matrix_size[1]))
    clear_screen()

    while True:
        
        operation = input("Выбор операции:\n"
                        "1. Сложение и вычитание матриц.\n"
                        "2. Умножение матрицы на вектор.\n"
                        "3. Умножение матриц.\n"
                        "4. Транспонирование матрицы.\n"
                        "5. Нахождение определителя (для квадратной матрицы).\n"
                        "6. Решение системы линейных уравнений методом Гаусса (если матрица квадратная).\n"
                        "7. Вернуться назад.\n"
                        "Выберите операцию: ")
        
        clear_screen()

        if operation == "1":
            while True:
                choice_operation = input("Выберите, какое действие выполнить: \n 1. Сложение \n 2. Вычитание \n Выберите действие: ")
                clear_screen()
                if choice_operation == "1":
                    print(f"Сумма матриц: {first_matrix.matrix} и {second_matrix.matrix} равна {first_matrix.add(second_matrix)}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break
                elif choice_operation == "2":
                    choice_vector = input(f"Выберите, какое действие выполнить:\n"
                                          f"1. Вычитание из матрицы {first_matrix.matrix} матрицу {second_matrix.matrix}\n"
                                          f"2. Вычитание из матрицы {second_matrix.matrix} матрицу {first_matrix.matrix}\n"
                                          f"Выберите действие: ")

                    clear_screen()
                    while True:
                        if choice_vector == "1":
                            print(f"Результат вычисления: {first_matrix.subtract(second_matrix)}")
                            input("Нажмите Enter для продолжения...")
                            clear_screen()
                            break
                        elif choice_vector == "2":
                            print(f"Результат вычисления: {second_matrix.subtract(first_matrix)}")
                            input("Нажмите Enter для продолжения...")
                            clear_screen()
                            break
                        else:
                            clear_screen()
                            print("Некорректный выбор. Попробуйте снова")
                    break
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "2":

            def validate_vector_multiply_input(matrix_size):
                while True:
                    try:
                        input_str = input(f"Введите координаты вектора ({matrix_size[1]} элементов через пробел): ")

                        if not input_str.strip():
                            raise ValueError("Ошибка: вектор не может быть пустым.")
                        
                        vector = []

                        for item in input_str.split():
                            try:
                                vector.append(float(item))  # Пробуем преобразовать каждый элемент в float
                            except ValueError:
                                raise ValueError(f"Ошибка: '{item}' не является числом.")

                        if len(vector) != matrix_size[1]:
                            raise ValueError(f"Ошибка: вектор должен содержать {matrix_size[1]} элемента(ов)")
                        clear_screen()
                        return vector  # Корректный ввод

                    except ValueError as e:
                        clear_screen()
                        print(e)

            while True:
 
                choice_matrix = input(f"Выберите матрицу для умножения: \n 1. {first_matrix.matrix} \n 2. {second_matrix.matrix} \nВыберите матрицу: ")
                clear_screen()
            
                if choice_matrix == "1":
                
                    print(f"Результат вычисления: {first_matrix.multiply_by_vector(validate_vector_multiply_input(first_matrix_size))}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break

                elif choice_matrix == "2":

                    print(f"Результат вычисления: {second_matrix.multiply_by_vector(validate_vector_multiply_input(second_matrix_size))}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break
                
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "3":
            print(f"Результат умножения матриц: {first_matrix.matrix} и {second_matrix.matrix} равен {first_matrix.multiply_by_matrix(second_matrix)}")
            input("Нажмите Enter для продолжения...")
            clear_screen()

        elif operation == "4":
            while True:
                choice_matrix = input(f"Выбор матрицы для транспонирования: \n 1. {first_matrix.matrix} \n 2. {second_matrix.matrix} \nВыберите матрицу: ")
                if choice_matrix == "1":
                    clear_screen()
                    print(f"Результат: {first_matrix.transpose()}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break  

                elif choice_matrix == "2":
                    clear_screen()
                    print(f"Результат: {second_matrix.transpose()}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break  

                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")
                    clear_screen()

        elif operation == "5":
            while True:
                choice_matrix = input(f"Выбор матрицы для нахождения определителя: \n 1. {first_matrix.matrix} \n 2. {second_matrix.matrix} \nВыберите вектор: ")
                clear_screen()
                if choice_matrix == "1":
                    print(f"Результат: {first_matrix.determinant()}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break

                elif choice_matrix == "2":
                    print(f"Результат: {second_matrix.determinant()}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break  

                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "6":

            def validate_vector_solve_gaussian_input(matrix_size):
                while True:
                    try:
                        input_str = input(f"Введите координаты вектора ({matrix_size[0]} элементов через пробел): ")

                        if not input_str.strip():  # Проверка на пустой ввод
                            raise ValueError("Ошибка: вектор не может быть пустым.")
                        
                        vector = []

                        # Преобразуем каждый элемент в float и обрабатываем возможные ошибки
                        for item in input_str.split():
                            try:
                                vector.append(float(item))  # Преобразуем каждый элемент в float
                            except ValueError:
                                raise ValueError(f"Ошибка: '{item}' не является числом.")
                        
                        # Проверяем, что количество элементов вектора соответствует ожидаемому
                        if len(vector) != matrix_size[0]:
                            raise ValueError(f"Ошибка: вектор должен содержать {matrix_size[0]} элемента(ов)")

                        return vector  # Корректный ввод

                    except ValueError as e:
                        print(e)
            
            while True:
 
                choice_matrix = input(f"Выберите матрицу для решения системы линейных уравнений методом Гаусса: \n 1. {first_matrix.matrix} \n 2. {second_matrix.matrix} \nВыберите матрицу: ")
                clear_screen()
            
                if choice_matrix == "1":
                    print(f"Результат вычисления: {first_matrix.solve_gaussian(validate_vector_solve_gaussian_input(first_matrix_size))}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break

                elif choice_matrix == "2":
                    vector = validate_vector_solve_gaussian_input(second_matrix_size)
                    clear_screen()  # Очищаем экран после успешного ввода
                    print(f"Результат вычисления: {second_matrix.solve_gaussian(validate_vector_solve_gaussian_input(second_matrix_size))}")
                    input("Нажмите Enter для продолжения...")
                    clear_screen()
                    break
                
                else:
                    clear_screen()
                    print("Некорректный выбор. Попробуйте снова")

        elif operation == "7":
            break

        else:
            clear_screen()
            print("Некорректный выбор. Попробуйте снова")

def equation_solver_menu():
    clear_screen()
    while True:
        
        choice_equation = input(f"Выбор типа уравнения: \n 1. Линейное. \n 2. Квадратное. \n 3. Вернуться назад.\nВыберите тип: ")
        
        if choice_equation == "1":
            def validate_linear_equation_input():
                clear_screen()
                while True:
                    
                    try:
                        list_coefficients = input("Введите коэффициенты линейного уравнения ax + b = 0 (a и b через пробел): ").split()
                        if len(list_coefficients) != 2:
                            raise ValueError("Укажите 2 коэффициента для линейного уравнения.")

                        # Преобразуем в числа с проверкой на ошибки
                        try:
                            list_coefficients = [float(el) for el in list_coefficients]

                        except ValueError:
                            raise ValueError("Все элементы должны быть числами.")

                        return dict(zip(["a", "b"], list_coefficients))

                    except ValueError as e:
                        clear_screen()
                        print(f"Ошибка: {e} Попробуйте снова.")
    
            linear_equation = EquationSolver(validate_linear_equation_input())
            clear_screen()
            print(f"Решение линейного уравнения: {linear_equation.solve_linear()}")
            input("Нажмите Enter для продолжения...")
            clear_screen()
            
        elif choice_equation == "2":
            def validate_quadratic_equation_input():
                clear_screen()
                while True:
                    try:
                        list_coefficients = input("Введите коэффициенты квадратного уравнения ax^2 + bx + c = 0 (a, b, c через пробел): ").split()
                        if len(list_coefficients) != 3:
                            raise ValueError("Укажите 3 коэффициента для квадратного уравнения.")

                        try:
                            list_coefficients = [float(el) for el in list_coefficients]
                        except ValueError:
                            raise ValueError("Все элементы должны быть числами.")
                        
                        return dict(zip(["a", "b", "c"], list_coefficients))

                    except ValueError as e:
                        clear_screen()
                        print(f"Ошибка: {e} Попробуйте снова.")
            
            coefficients = validate_quadratic_equation_input()
            clear_screen()
            print("Доступные параметры:")
            print("1. Только действительные корни")
            print("2. Только положительные корни")
            print("3. Без условий")
            
            while True:
                add_parameters = input("Выберите параметры: 1, 2 или 3: ")
                if add_parameters == "1":
                    parameters = {'condition': 'real roots only'}
                    break
                elif add_parameters == "2":
                    parameters = {'condition': 'positive roots only'}
                    break
                elif add_parameters == "3" or add_parameters == "no":
                    parameters = {'condition': 'no condition'}
                    break
                else:
                    clear_screen()
                    print("Некорректный ввод. Пожалуйста, выберите 1, 2 или 3.")

            clear_screen()
            quadratic_equation = EquationSolver(coefficients, parameters)
            print(f"Решение квадратного уравнения: {quadratic_equation.solve_quadratic()}")
            input("Нажмите Enter для продолжения...")
            clear_screen()

        elif choice_equation == "3":
            break

        else:
            clear_screen()
            print("Некорректный выбор. Попробуйте снова")

main_menu()