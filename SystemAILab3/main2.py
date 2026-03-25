from rdflib import Graph, Namespace, Literal, URIRef
import skfuzzy as fuzz
import numpy as np


EX = Namespace("http://www.semanticweb.org/vyvy07/ontologies/2026/2/StreetLight#")

# Загрузка онтологии
g = Graph()
g.parse("street_light2.rdf")

# Функция для получения правил из онтологии
def get_rules():
    rules = []
    # Ищем все индивидуалы типа Rule
    for rule in g.subjects(predicate=URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type"), object=EX.Rule):
        rule_name = str(rule)
        conditions = []
        action = None

        # Получаем условия, связанные с правилом
        for cond in g.objects(subject=rule, predicate=EX.hasCondition):
            cond_dict = {}
            # Получаем свойства условия
            for prop, value in g.predicate_objects(subject=cond):
                prop_str = str(prop)
                val_str = str(value)
                if prop_str.endswith("hasLightLevel"):
                    cond_dict['lightLevel'] = float(val_str)
                elif prop_str.endswith("hasTimeofDay"):
                    cond_dict['timeOfDay'] = val_str
                elif prop_str.endswith("hasWeather"):
                    cond_dict['weather'] = val_str
            conditions.append(cond_dict)

        # Получаем действие, связанное с правилом
        for act in g.objects(subject=rule, predicate=EX.hasAction):
            # предполагается, что действие — это индивидуал
            action = str(act)

        rules.append({'name': rule_name, 'conditions': conditions, 'action': action})
    return rules

# Получение текущих условий (пример)
def get_current_conditions():
    # Здесь можно реализовать ввод с датчиков или пользовательский ввод
    return {
        'lightLevel': 20,
        'timeOfDay': 'Night',
        'weather': 'Rain'
    }

# Фаззификация
def fuzzyfy_conditions(conditions):
    # Создаем массив x для светового уровня
    x_light = np.linspace(0, 100, 1000)  # диапазон светового уровня

    # Фаззификация светового уровня
    low_light_mf = fuzz.trimf(x_light, [0, 0, 50])
    high_light_mf = fuzz.trimf(x_light, [50, 100, 100])
    light_level = conditions['lightLevel']
    degree_low = fuzz.interp_membership(x_light, low_light_mf, light_level)
    degree_high = fuzz.interp_membership(x_light, high_light_mf, light_level)

    # Для времени суток
    time_of_day = 1 if conditions['timeOfDay'] == 'Night' else 0

    # Для погоды
    weather = 1 if conditions['weather'] == 'Rain' else 0

    return {
        'light_low': degree_low,
        'light_high': degree_high,
        'timeNight': time_of_day,
        'weatherRain': weather
    }

# Дефаззификация
def defuzzify(output_value):
    if output_value > 0.5:
        return 'TurnOn'
    elif output_value < 0.3:
        return 'TurnOff'
    else:
        return 'Dim'

# Основная логика
def main():
    rules = get_rules()
    current_conditions = get_current_conditions()
    fuzzy_conditions = fuzzyfy_conditions(current_conditions)

    print(f"Current conditions: {current_conditions}")
    print(f"Fuzzy conditions: {fuzzy_conditions}")
    print("Rules from ontology:")
    for r in rules:
        print(r)

    # Обработка правил
    decision_values = []
    for rule in rules:
        conds = rule['conditions']
        # В случае отсутствия условий по конкретному свойству — пропускаем или задаем по умолчанию
        # Для этого можно расширить

        # Получаем параметры условий правила
        rule_light_level = conds[0].get('lightLevel', 50)
        rule_time_str = conds[0].get('timeOfDay', 'Day')
        rule_weather_str = conds[0].get('weather', 'Clear')

        rule_time = 1 if rule_time_str == 'Night' else 0
        rule_weather = 1 if rule_weather_str == 'Rain' else 0

        # Вычисляем степень совпадения с текущими условиями
        # Используем интерполяцию
        # Для света используем интерполяцию по диапазону
        x_light = np.linspace(0, 100, 1000)
        low_mf = fuzz.trimf(x_light, [0, 0, 50])
        high_mf = fuzz.trimf(x_light, [50, 100, 100])

        light_match_low = fuzz.interp_membership(x_light, low_mf, current_conditions['lightLevel'])
        light_match_high = fuzz.interp_membership(x_light, high_mf, current_conditions['lightLevel'])

        # Для условий времени и погоды - простое сравнение
        time_match = 1 if fuzzy_conditions['timeNight'] == rule_time else 0
        weather_match = 1 if fuzzy_conditions['weatherRain'] == rule_weather else 0

        # Объединяем условия (например, минимум)
        rule_strength_low = min(light_match_low, time_match, weather_match)
        rule_strength_high = min(light_match_high, time_match, weather_match)

        # Для примера возьмем максимальную из двух
        rule_strength = max(rule_strength_low, rule_strength_high)

        decision_values.append((rule['action'], rule_strength))

    # Находим наиболее сильное правило
    best_action = None
    max_strength = -1
    for action, strength in decision_values:
        if strength > max_strength:
            max_strength = strength
            best_action = action

    # Дефаззификация — определить действие
    final_action = defuzzify(max_strength)

    print(f"Decided action: {final_action} (from rule: {best_action})")


if __name__ == "__main__":
    main()