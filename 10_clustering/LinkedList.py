import numpy as np

class LinkedList:
    class Node:
        def __init__(self, value, prev=None, next=None):
            self.value = value
            self.next = next
            self.prev = prev

    def __init__(self, dtype=None):
        self.head = None
        self.tail = None
        self.dtype = dtype
        self.size = 0

    def init_dtype(self, value):
        try:
            int(value)
            self.dtype = type(value)
        except:
            raise Exception("Cannot initialize list with this dtype, ONLY NUMBERS ARE ADMITTED")

    def push_front(self, value):
        if self.dtype is None:
            self.init_dtype(value)
        else:
            if type(value) != self.dtype:
                raise Exception("The type of the value does not match with the type of the list")
        if self.size == 0:
            self.head = self.tail = self.Node(value=value)
        else:
            self.head.prev = self.head = self.Node(value=value, next=self.head)
        self.size += 1

    def push_back(self, value):
        if self.dtype is None:
            self.init_dtype(value)
        else:
            if type(value) != self.dtype:
                raise Exception("The type of the value does not match with the type of the list")
        if self.size == 0:
            self.head = self.tail = self.Node(value=value)
        else:
            self.tail.next = self.tail = self.Node(value=value, prev=self.tail)
        self.size += 1

    def pop_front(self):
        if self.size == 0:
            raise Exception("POP_FRONT not possible: list is EMPTY")
        else:
            tmp = self.head
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            else:
                self.head.prev = None
            self.size -= 1
            tmp.next = tmp.prev = None
            return tmp.value

    def pop_back(self):
        if self.size == 0:
            raise Exception("POP_BACK not possible: list is EMPTY")
        else:
            tmp = self.tail
            self.tail = self.tail.prev
            if self.tail is None:
                self.head = None
            else:
                self.tail.next = None
            self.size -= 1
            tmp.next = tmp.prev = None
            return tmp.value
    
    def print_forward(self):
        if self.size == 0:
            print("EMPTY LIST")
        else:
            i = self.head
            while i is not None:
                print(i.value, " ", end="")
                i = i.next
            print()
    
    def print_backward(self):
        if self.size == 0:
            print("EMPTY LIST")
        else:
            i = self.tail
            while i is not None:
                print(i.value, " ", end="")
                i = i.prev
            print()
    
    def get(self, k):
        if self.size <= k:
            raise Exception("Index out of bound")
        else:
            if k <= int(self.size/2):
                node = self.head
                for _ in range(k):
                    node = node.next
            else:
                node = self.tail
                for _ in range(self.size - k - 1):
                    node = node.prev
            return node.value
    
    def to_array(self):
        if self.size == 0:
            return np.array([])
        else:
            list_as_array = np.zeros(self.size, dtype=self.dtype)
            node = self.head
            i = 0
            while node is not None:
                list_as_array[i] = node.value
                node = node.next
                i += 1
            return list_as_array
    
    def to_array_reverse(self):
        if self.size == 0:
            return np.array([])
        else:
            list_as_array = np.zeros(self.size, dtype=self.dtype)
            node = self.tail
            i = 0
            while node is not None:
                list_as_array[i] = node.value
                node = node.prev
                i += 1
            return list_as_array
