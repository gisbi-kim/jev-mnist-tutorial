"""Pure task definitions. No model, filesystem or network dependencies."""
import re

ACTIONS = ('-1', '0', '+1')
EN = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
KO = ('영', '일', '이', '삼', '사', '오', '육', '칠', '팔', '구')


def oracle_action(target: int, current: int) -> str:
    return '+1' if target > current else '-1' if target < current else '0'


def apply_action(current: int, action: str) -> tuple[int, bool]:
    if action not in ACTIONS:
        raise ValueError('Invalid action')
    return current + int(action), action == '0'


def questions() -> dict:
    return {'move': {'type': 'choice', 'instructions':
        'Read the handwritten target digit in the image. Choose exactly one operation '
        'that moves the current number toward it. Choose 0 only when equal.',
        'criteria': {'-1': 'Subtract one from the current number.',
                     '0': 'Keep the current number and STOP: already equal to the image digit.',
                     '+1': 'Add one to the current number.'}}}


def legacy_state_text(current: int, variant: int = 0) -> str:
    """Never insert target labels into the model input."""
    forms = (f'The current number is {current}.', f'현재 숫자는 {current}이야.',
             f'Start from {EN[current] if 0 <= current <= 9 else current}.',
             f'{KO[current] if 0 <= current <= 9 else current}에서 시작해.',
             f'My counter now reads {current}.', f'지금 값은 {current}. 다음 연산을 골라줘.')
    return forms[variant % len(forms)]


def state_text(current: int, variant: int = 0) -> str:
    """All newly generated model instructions are English."""
    forms = (f'The current number is {current}.',
             f'My counter now reads {current}.',
             f'Start from {EN[current] if 0 <= current <= 9 else current}.',
             f'The current value is {current}.',
             f'The counter is currently at {current}.',
             f'Begin with the number {current}.')
    return forms[variant % len(forms)]


def parse_start(text: str) -> int:
    """Restricted natural-language parser for UI state, NOT an image recognizer."""
    if not isinstance(text, str) or len(text) > 200:
        raise ValueError('시작 지시는 200자 이내로 입력해줘.')
    if re.search(r'\d\.\d|(?<!\d)\.\d', text):
        raise ValueError('정수 0~9 중 하나를 입력해줘.')
    numbers = re.findall(r'(?<![\w.])[+-]?\d+(?![\w.]|\.\d)', text)
    # Korean particles follow a number directly, e.g. 2에서 시작해.
    if not numbers:
        numbers = re.findall(r'(?<![\d.])[+-]?\d+(?![\d.])', text)
    if len(numbers) == 1:
        value = int(numbers[0])
    elif numbers:
        raise ValueError('시작 숫자는 하나만 적어줘.')
    else:
        found = [i for i, word in enumerate(EN) if re.search(r'\b' + word + r'\b', text.lower())]
        korean = re.fullmatch(r'\s*([영공일이삼사오육칠팔구])(?:에서\s*시작(?:해|하자)?|부터\s*시작(?:해|하자)?)?[.!?]?\s*', text)
        if korean:
            found.append(0 if korean[1] == '공' else KO.index(korean[1]))
        if len(found) != 1:
            raise ValueError('예: “2에서 시작해”, “삼에서 시작해”, “start from seven”.')
        value = found[0]
    if not 0 <= value <= 9:
        raise ValueError('이 실험의 시작 숫자는 0~9야.')
    return value


def rollout_summary(target: int, start: int, policy, max_turns: int = 20) -> dict:
    current, trace, visited = start, [], set()
    for _ in range(max_turns):
        if current in visited:
            return {'success': False, 'reason': 'cycle', 'trace': trace, 'extra_moves': None}
        visited.add(current)
        action = policy(current)
        after, stopped = apply_action(current, action)
        trace.append({'before': current, 'action': action, 'after': after})
        if stopped:
            success = after == target
            return {'success': success, 'reason': 'success' if success else 'wrong_stop',
                    'trace': trace, 'extra_moves': len(trace) - 1 - abs(start-target) if success else None}
        current = after
        if current < 0 or current > 9:
            return {'success': False, 'reason': 'out_of_range', 'trace': trace, 'extra_moves': None}
    return {'success': False, 'reason': 'turn_limit', 'trace': trace, 'extra_moves': None}
