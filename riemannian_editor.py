import tkinter as tk
from tkinter import filedialog, colorchooser
import numpy as np
from PIL import Image, ImageTk

class RiemannianImageEditor:
    """Главный класс редактора римановых изображений.
    
    Обеспечивает интерфейс для визуализации и редактирования изображений 
    с учетом римановой метрики. Позволяет:
    - Перемещаться по изображению с учетом метрики
    - Изменять цвета пикселей
    - Сохранять изменения
    - Настраивать параметры отображения
    
    Attributes:
        parent: Родительское окно Tkinter
        window: Окно редактора
        color_array: NumPy массив с цветами изображения
        metric_array: NumPy массив с метрикой изображения
        color_file_path: Путь к файлу с цветами
        metric_file_path: Путь к файлу с метрикой
        position: Текущая позиция просмотра
        right_dir: Вектор направления вправо
        up_dir: Вектор направления вверх
        selected_color: Выбранный цвет для рисования (RGB)
        computation_resolution: Разрешение вычислений
        move_speed: Скорость перемещения
        rotation_speed: Скорость вращения
        scale_speed: Скорость масштабирования
        integration_step: Шаг интегрирования
        canvas_width: Ширина холста
        canvas_height: Высота холста
    """
    
    __slots__ = ("parent", "window", "color_array", "metric_array", "color_file_path", 
                "metric_file_path", "position", "right_dir", "up_dir", "selected_color", 
                "computation_resolution", "move_speed", "rotation_speed", "scale_speed", 
                "integration_step", "canvas_width", "canvas_height", "canvas", 
                "btn_choose_color", "btn_save", "resolution_scale", "integration_step_scale", 
                "btn_close", "display_image", "tk_image")

    def __init__(self, parent, color_array, metric_array, color_file_path, metric_file_path):
        """Инициализация редактора изображений.
        
        Args:
            parent: Родительское окно Tkinter
            color_array: NumPy массив с цветами изображения (H×W×3)
            metric_array: NumPy массив с метрикой изображения (H×W×3)
            color_file_path: Путь к файлу с цветами
            metric_file_path: Путь к файлу с метрикой
        """
        self.parent = parent
        self.window = tk.Toplevel(parent)
        self.window.title("Редактор рисунков")
        
        # Сохраняем данные
        self.color_array = color_array
        self.metric_array = metric_array
        self.color_file_path = color_file_path
        self.metric_file_path = metric_file_path
        
        # Параметры просмотра по умолчанию
        self.position = np.array([self.color_array.shape[0] // 2, self.color_array.shape[1] // 2], dtype=np.float32)
        self.right_dir = np.array([20.0, 0.0], dtype=np.float32)
        self.up_dir = np.array([0.0, 20.0], dtype=np.float32)
        self.right_dir_from_up_dir()
        self.selected_color = (255, 0, 0)
        self.computation_resolution = 36
        self.move_speed = 0.1
        self.rotation_speed = 0.1
        self.scale_speed = 0.1
        self.integration_step = 10

        # Создаём интерфейс
        self.create_widgets()

    def create_widgets(self):
        """Создает интерфейс редактора (холст и элементы управления)."""
        # Холст для отображения
        self.canvas_width = 600
        self.canvas_height = 600
        self.canvas = tk.Canvas(self.window, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack()

        # Панель управления
        control_frame = tk.Frame(self.window)
        control_frame.pack(pady=10)

        self.btn_choose_color = tk.Button(control_frame, text="Выбрать цвет", command=self.choose_color)
        self.btn_choose_color.pack(side=tk.LEFT, padx=5)
        
        self.btn_save = tk.Button(control_frame, text="Сохранить изменения", command=self.save_changes)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.resolution_scale = tk.Scale(control_frame, from_=8, to=64, resolution=1, orient=tk.HORIZONTAL, command=self.choose_resolution)
        self.resolution_scale.set(self.computation_resolution)
        self.resolution_scale.pack(side=tk.LEFT, padx=5)

        self.integration_step_scale = tk.Scale(control_frame, from_=1, to=100, resolution=1, orient=tk.HORIZONTAL, command=self.choose_integration_step)
        self.integration_step_scale.set(self.integration_step)
        self.integration_step_scale.pack(side=tk.LEFT, padx=5)
        
        self.btn_close = tk.Button(control_frame, text="Закрыть", command=self.window.destroy)
        self.btn_close.pack(side=tk.LEFT, padx=5)
        
        # Привязка событий мыши
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)

        # Привязка событий клавиатуры
        self.window.bind('<KeyPress>', self.on_key_press)
        self.window.focus_set()
        
        # Первоначальная отрисовка
        self.redraw_canvas()
    
    def exponential_mapping(self, v):
        """Вычисляет экспоненциальное отображение вектора в касательном пространстве.
        
        Args:
            v: Вектор в касательном пространстве (NumPy array [x, y])
            
        Returns:
            NumPy array: Координаты точки на изображении после экспоненциального отображения
        """
        vel0 = float(v[0])
        vel1 = float(v[1])

        pos = self.position.copy()

        inv_step = 1.0 / float(self.integration_step)

        for _ in range(self.integration_step):
            x, y = int(pos[0]), int(pos[1])

            # Получаем метрику и соседние значения
            g = self.metric_array[x, y]
            g_right = self.metric_array[min(x+1, self.metric_array.shape[0]-1), y]
            g_left = self.metric_array[max(x-1, 0), y]
            g_top = self.metric_array[x, min(y+1, self.metric_array.shape[1]-1)]
            g_bottom = self.metric_array[x, max(y-1, 0)]

            # Вычисляем производные
            g_x = g_right - g_left
            g_y = g_top - g_bottom

            christoff_x0 = g_x[0]
            christoff_x1 = 2 * g_x[1] - g_y[0]
            
            christoff_xy0 = g_y[0]
            christoff_xy1 = g_x[2]
            
            christoff_y0 = 2 * g_y[1] - g_x[2]
            christoff_y1 = g_y[2]

            v0, v1 = vel0, vel1
            v0_sq = v0 * v0
            v1_sq = v1 * v1
            v0_v1 = 2 * v0 * v1

            acc0 = christoff_x0 * v0_sq + christoff_xy0 * v0_v1 + christoff_y0 * v1_sq
            acc1 = christoff_x1 * v0_sq + christoff_xy1 * v0_v1 + christoff_y1 * v1_sq
        
            inv_det_g = 0.125 / (g[0] * g[2] - g[1]**2)

            acc0 *= inv_det_g
            acc1 *= inv_det_g
            
            tmp0 = g[1] * acc1 - g[2] * acc0
            tmp1 = g[1] * acc0 - g[0] * acc1
            
            acc0, acc1 = tmp0 * inv_step, tmp1 * inv_step

            vel0 += acc0
            vel1 += acc1
            pos[0] += vel0 * inv_step
            pos[1] += vel1 * inv_step
            vel0 += acc0
            vel1 += acc1

            if pos[0] < 0 or pos[0] >= self.metric_array.shape[0] or pos[1] < 0 or pos[1] >= self.metric_array.shape[1]:
                pos[0] = min(max(0, pos[0]), self.metric_array.shape[0] - 1)
                pos[1] = min(max(0, pos[1]), self.metric_array.shape[1] - 1)
                break

        return pos
    
    def on_key_press(self, event):
        """Обработчик нажатий клавиш для управления редактором.
        
        Args:
            event: Событие клавиатуры Tkinter
        """
        if event.keysym.lower() == 'w':
            self.position = self.exponential_mapping(self.up_dir * (-self.move_speed))
        elif event.keysym.lower() == 's':
            self.position = self.exponential_mapping(self.up_dir * self.move_speed)
        elif event.keysym.lower() == 'd':
            self.position = self.exponential_mapping(self.right_dir * self.move_speed)
        elif event.keysym.lower() == 'a':
            self.position = self.exponential_mapping(self.right_dir * (-self.move_speed))
        elif event.keysym == 'Left':
            self.rotate_directions(self.rotation_speed)
        elif event.keysym == 'Right':
            self.rotate_directions(-self.rotation_speed)

        self.update_position()
        self.right_dir_from_up_dir()
        
        # Постдвиженческая отрисовка
        self.redraw_canvas()

    def update_position(self):
        """Ограничивает позицию просмотра границами изображения."""
        if 0 > self.position[0]:
            self.position[0] = 0.0
        elif self.position[0] >= self.metric_array.shape[0]:
            self.position[0] = float(self.metric_array.shape[0] - 1)
        
        if 0 > self.position[1]:
            self.position[1] = 0.0
        elif self.position[1] >= self.metric_array.shape[1]:
            self.position[1] = float(self.metric_array.shape[1] - 1)

    def get_l2_from_metric(self, x, y, g_x, g_xy, g_y):
        """Вычисляет квадрат длины вектора по метрике.
        
        Args:
            x: Координата x вектора
            y: Координата y вектора
            g_x: Компонента метрики g_11
            g_xy: Компонента метрики g_12
            g_y: Компонента метрики g_22
            
        Returns:
            float: Квадрат длины вектора
        """
        return g_x * (x**2) + 2 * g_xy * x * y + g_y * (y**2)
    
    def get_metric(self, x0, y0, x1, y1, g_x, g_xy, g_y):
        """Вычисляет значение метрики для двух векторов.
        
        Args:
            x0, y0: Координаты первого вектора
            x1, y1: Координаты второго вектора
            g_x: Компонента метрики g_11
            g_xy: Компонента метрики g_12
            g_y: Компонента метрики g_22
            
        Returns:
            float: Значение метрики
        """
        return g_x * x0 * x1 + g_xy * (x0 * y1 + x1 * y0) + g_y * y0 * y1
    
    def get_l2(self, x, y):
        """Вычисляет квадрат длины вектора в текущей позиции.
        
        Args:
            x: Координата x вектора
            y: Координата y вектора
            
        Returns:
            float: Квадрат длины вектора
        """
        g = self.metric_array[int(self.position[0]), int(self.position[1])]
        return self.get_l2_from_metric(x, y, g[0], g[1], g[2])
    
    def right_dir_from_up_dir(self):
        """Вычисляет ортогональное направление вправо из направления вверх."""
        self.update_position()

        g = self.metric_array[int(self.position[0]), int(self.position[1])]

        g_right = self.get_l2_from_metric(self.right_dir[0], self.right_dir[1], g[0], g[1], g[2])
        g_up = self.get_l2_from_metric(self.up_dir[0], self.up_dir[1], g[0], g[1], g[2])
        g_right_up = self.get_metric(self.up_dir[0], self.up_dir[1], self.right_dir[0], self.right_dir[1], g[0], g[1], g[2])
        g_new_right = g_right - (g_right_up ** 2) / g_up

        new_right_x = self.right_dir[0] - self.up_dir[0] * g_right_up / g_up
        new_right_y = self.right_dir[1] - self.up_dir[1] * g_right_up / g_up

        right_factor = np.sqrt(g_up / g_new_right)

        self.right_dir[0] = new_right_x * right_factor
        self.right_dir[1] = new_right_y * right_factor

    def rotate_directions(self, angle):
        """Поворачивает направления просмотра на заданный угол.
        
        Args:
            angle: Угол поворота в радианах
        """
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        new_x = self.up_dir[0]*cos_a + self.up_dir[1]*sin_a
        new_y = self.up_dir[1]*cos_a - self.up_dir[0]*sin_a

        self.up_dir = np.array([new_x, new_y])*np.sqrt(self.get_l2(self.up_dir[0], self.up_dir[1])/self.get_l2(new_x, new_y))

        self.right_dir_from_up_dir()

    def choose_color(self):
        """Открывает диалог выбора цвета для рисования."""
        color = colorchooser.askcolor(title="Выбор цвета")
        if color[0] is not None:
            self.selected_color = tuple(map(int, color[0]))
    
    def save_changes(self):
        """Сохраняет изменения в файлы цветов и метрики."""
        if self.color_file_path and self.metric_file_path:
            try:
                img = Image.fromarray(self.color_array)
                img.save(self.color_file_path)
                print("Изменения цвета сохранены в", self.color_file_path)
            except Exception as e:
                print("Ошибка сохранения изменений цвета:", e)
            try:
                img = Image.fromarray((self.metric_array*255).astype(np.uint8))
                img.save(self.metric_file_path)
                print("Изменения метрики сохранены в", self.metric_file_path)
            except Exception as e:
                print("Ошибка сохранения изменений метрики:", e)
    
    def choose_resolution(self, value):
        """Изменяет разрешение вычислений.
        
        Args:
            value: Новое значение разрешения
        """
        self.computation_resolution = int(value)
        self.redraw_canvas()
    
    def choose_integration_step(self, value):
        """Изменяет шаг интегрирования.
        
        Args:
            value: Новое значение шага интегрирования
        """
        self.integration_step = int(value)
        self.redraw_canvas()
    
    def redraw_canvas(self):
        """Перерисовывает холст с текущими параметрами."""
        if not hasattr(self, 'canvas'):
            return
        
        # Создаем изображение для отображения
        compute_img = np.zeros((self.computation_resolution, self.computation_resolution, 3), dtype=np.uint8)
        
        center = self.computation_resolution // 2
        
        for y in range(self.computation_resolution):
            for x in range(self.computation_resolution):
                nx = (x - center) / float(self.computation_resolution)
                ny = (y - center) / float(self.computation_resolution)
                
                # Вектор в касательном пространстве
                v = self.right_dir * nx + self.up_dir * ny
                
                # Экспоненциальное отображение
                point = self.exponential_mapping(v).astype(int)
                
                # Проверка границ
                if (0 <= point[0] < self.color_array.shape[0] and 
                    0 <= point[1] < self.color_array.shape[1]):
                    compute_img[y, x] = self.color_array[point[0], point[1]]
                else:
                    compute_img[y, x] = [0, 0, 0]
        
        # Преобразуем массив в изображение для tkinter
        self.display_image = Image.fromarray(compute_img)
        self.display_image = self.display_image.resize((self.canvas_width, self.canvas_height), Image.Resampling.NEAREST)
        self.tk_image = ImageTk.PhotoImage(self.display_image)
        
        # Обновляем холст
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)
    
    def on_canvas_drag(self, event):
        """Обработчик перетаскивания мыши для рисования.
        
        Args:
            event: Событие мыши Tkinter
        """
        if event.widget != self.canvas:
            return
        
        center_x = self.canvas_width // 2
        center_y = self.canvas_height // 2
        
        # Нормированные координаты относительно центра
        nx = (event.x - center_x) / float(self.canvas_width)
        ny = (event.y - center_y) / float(self.canvas_height)
        
        # Вектор в касательном пространстве
        v = self.right_dir * nx + self.up_dir * ny
        
        # Экспоненциальное отображение
        point = self.exponential_mapping(v).astype(int)
        
        # Проверка границ и изменение цвета
        if (0 <= point[0] < self.color_array.shape[0] and 
            0 <= point[1] < self.color_array.shape[1]):
            self.color_array[point[0], point[1]] = self.selected_color
            self.redraw_canvas()
    
    def on_mouse_wheel(self, event):
        """Обработчик колеса мыши для масштабирования.
        
        Args:
            event: Событие мыши Tkinter
        """
        # Масштабирование при прокрутке колеса мыши
        if event.delta > 0:
            self.up_dir *= (1+self.scale_speed)
            self.right_dir *= (1+self.scale_speed)
        else:
            self.up_dir *= (1-self.scale_speed)
            self.right_dir *= (1-self.scale_speed)
        
        # Переотрисовка после движений
        self.redraw_canvas()


class RiemannianImageEditorMainMenu:
    """Главное меню редактора римановых изображений.
    
    Обеспечивает интерфейс для загрузки/создания изображений 
    перед открытием основного редактора.
    
    Attributes:
        root: Главное окно Tkinter
        current_color_array: Загруженный массив цветов
        current_metric_array: Загруженный массив метрики
        current_color_file_path: Путь к файлу цветов
        current_metric_file_path: Путь к файлу метрики
    """
    
    def __init__(self, root):
        """Инициализация главного меню.
        
        Args:
            root: Главное окно Tkinter
        """
        self.root = root
        self.root.title("Выбор рисунков")
        self.create_widgets()
    
    def create_widgets(self):
        """Создает интерфейс главного меню."""
        # Кнопки главного меню
        self.btn_load_color = tk.Button(self.root, text="Загрузить поле цветов(BMP)", command=self.load_color_bmp)
        self.btn_load_color.pack(pady=10)
        
        self.btn_load_metric = tk.Button(self.root, text="Загрузить поле метрики(BMP)", command=self.load_metric_bmp)
        self.btn_load_metric.pack(pady=10)

        self.btn_create_new = tk.Button(self.root, text="Создать новые(BMP)", command=self.create_new_bmps)
        self.btn_create_new.pack(pady=10)
        
        self.btn_open_editor = tk.Button(self.root, text="Открыть редактор", command=self.open_editor, state=tk.DISABLED)
        self.btn_open_editor.pack(pady=10)
        
        # Инициализация переменных
        self.current_color_array = None
        self.current_metric_array = None
        self.current_color_file_path = None
        self.current_metric_file_path = None
    
    def load_color_bmp(self):
        """Загружает файл с цветами изображения."""
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.current_color_file_path = file_path
            try:
                img = Image.open(self.current_color_file_path)
                self.current_color_array = np.array(img)
                print("Массив цветов загружен:", self.current_color_array.shape)
                self.check_buttons_state()
            except Exception as e:
                print("Ошибка загрузки файла цветов:", e)
    
    def load_metric_bmp(self):
        """Загружает файл с метрикой изображения.
        
        Проверяет, что метрика является римановой (положительно определенной).
        """
        file_path = filedialog.askopenfilename(filetypes=[("BMP files", "*.bmp")])
        if file_path:
            self.current_metric_file_path = file_path
            try:
                img = Image.open(self.current_metric_file_path)
                self.current_metric_array = np.array(img)
                if len(self.current_metric_array.shape) == 3 and self.current_metric_array.shape[2] >= 3:
                    self.current_metric_array = self.current_metric_array[:, :, :3].astype(np.float32)
                    is_riemannian = True
                    for y in range(self.current_metric_array.shape[1]):
                        for x in range(self.current_metric_array.shape[0]):
                            if not (self.current_metric_array[y, x][0] > 0 and (self.current_metric_array[y, x][0] * self.current_metric_array[y, x][2] > self.current_metric_array[y, x][1] ** 2)):
                                is_riemannian = False
                                break
                    if is_riemannian:
                        self.current_metric_array /= 255.0
                        print("Массив метрики загружен:", self.current_metric_array.shape)
                        self.check_buttons_state()
                    else:
                        print("Ошибка: Это файл не римановой метрики, а нужен римановой")
                        self.current_metric_array = None
                else:
                    print("Ошибка: Файл метрики обязан содержать 3 значения на пиксель")
            except Exception as e:
                print("Ошибка загрузки файла метрики:", e)
    
    def create_new_bmps(self):
        """Создает новые файлы цветов и метрики заданного размера."""
        try:
            size = tk.simpledialog.askinteger("Размер", "Введите размер:", parent=self.root, minvalue=16, maxvalue=128)
            if not size:
                return
            
            color_array = np.zeros((size, size, 3), dtype=np.uint8)
            metric_array = np.zeros((size, size, 3), dtype=np.uint8)
            metric_array[:] = [255, 0, 255]

            color_file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP files", "*.bmp")], title="Сохранить цвета (BMP) как:")
            if not color_file_path:
                return
            
            metric_file_path = filedialog.asksaveasfilename(defaultextension=".bmp", filetypes=[("BMP files", "*.bmp")], title="Сохранить метрику (BMP) как:")
            if not metric_file_path:
                return
            
            Image.fromarray(color_array).save(color_file_path)
            Image.fromarray(metric_array).save(metric_file_path)

            self.current_color_file = color_file_path
            self.current_metric_file = metric_file_path
            self.current_color_array = color_array
            self.current_metric_array = metric_array.astype(np.float32) / 255.0
            
            self.check_buttons_state()
            print("Выполнено: Новые цвет", self.current_color_file, "и метрика", self.current_metric_file, "созданы")
            
        except Exception as e:
            print("Ошибка создания:", e)

    def check_buttons_state(self):
        """Проверяет состояние кнопок в зависимости от загруженных данных."""
        if self.current_color_array is not None and self.current_metric_array is not None:
            self.btn_open_editor.config(state=tk.NORMAL)
        else:
            self.btn_open_editor.config(state=tk.DISABLED)
    
    def open_editor(self):
        """Открывает редактор с текущими загруженными данными."""
        if self.current_color_array is None or self.current_metric_array is None:
            return
        
        # Создаем новое окно редактора
        RiemannianImageEditor(self.root, self.current_color_array.copy(), self.current_metric_array.copy(), self.current_color_file_path, self.current_metric_file_path)
        
        # Очищаем меню
        self.current_color_array = None
        self.current_metric_array = None
        self.check_buttons_state()

if __name__ == "__main__":
    root = tk.Tk()
    app = RiemannianImageEditorMainMenu(root)
    root.mainloop()
