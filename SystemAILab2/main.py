import numpy as np
import matplotlib.pyplot as plt


def trapezoidal_membership(x, a, b, c, d):
    """
    Трапециевидная функция принадлежности.
    Возвращает степень принадлежности элемента x к нечёткому множеству.
    """
    if x <= a or x >= d:
        return 0.0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x < c:
        return 1.0
    else:  # c <= x < d
        return (d - x) / (d - c)


def get_membership_values(x_values, a, b, c, d):
    """Вычисляет значения функции принадлежности для массива x_values."""
    return np.array([trapezoidal_membership(x, a, b, c, d) for x in x_values])


def calculate_complement(membership_values):
    """Вычисляет дополнение нечёткого множества: 1 - μ(x)."""
    return 1 - membership_values


def plot_results(x_values, membership_values, complement_values, label):
    """Строит график исходной функции принадлежности и её дополнения."""
    plt.figure(figsize=(12, 7))

    plt.plot(x_values, membership_values, label=f'Исходная функция ({label})', color='blue', linewidth=2)
    plt.plot(x_values, complement_values, label=f'Дополнение ({label})', color='red', linewidth=2, linestyle='--')

    plt.title(f'Трапециевидная функция принадлежности для "{label}" и её дополнение', fontsize=14)
    plt.xlabel('Уровень освещённости (люкс)', fontsize=12)
    plt.ylabel('Степень принадлежности', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()
    plt.show()


def main():
    print("=== Управление освещением: анализ нечётких множеств ===\n")

    # Выбор категории для анализа
    print("Выберите категорию для анализа:")
    print("1. Освещённость помещения")
    print("2. Время суток")
    choice = input("Введите номер (1 или 2): ").strip()

    if choice == "1":
        category = "Освещённость помещения"
        subcategories = ["темно", "приглушённо", "ярко", "слишком ярко"]
    elif choice == "2":
        category = "Время суток"
        subcategories = ["утро", "день", "вечер", "ночь"]
    else:
        print("Неверный выбор. Завершение программы.")
        return

    print(f"\nВыбрана категория: {category}")
    print("Доступные подкатегории:", ", ".join(subcategories))
    subcategory = input("Выберите подкатегорию: ").strip()

    if subcategory not in subcategories:
        print("Неверная подкатегория. Завершение программы.")
        return

    # Ввод параметров трапеции
    print(f"\nВведите параметры трапециевидной функции принадлежности для '{subcategory}':")
    a = float(input("a (начало роста): "))
    b = float(input("b (начало плато): "))
    c = float(input("c (конец плато): "))
    d = float(input("d (конец спада): "))

    # Ввод чётких объектов (уровней освещённости в люксах)
    x_input = input("\nВведите чёткие объекты множества (уровни освещённости в люксах через пробел): ")
    x_values = np.array(list(map(float, x_input.split())))

    # Вычисление значений функции принадлежности
    membership_values = get_membership_values(x_values, a, b, c, d)

    # Вычисление дополнения
    complement_values = calculate_complement(membership_values)

    # Вывод результатов
    print(f"\nРезультаты для '{subcategory}' ({category}):")
    print("Чёткие объекты (люкс):", x_values)
    print("Степень принадлежности:   ", membership_values.round(3))
    print("Дополнение:            ", complement_values.round(3))

    # Построение графика
    plot_results(x_values, membership_values, complement_values, f"{subcategory} ({category})")


if __name__ == "__main__":
    main()
