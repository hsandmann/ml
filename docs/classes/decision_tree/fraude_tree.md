```mermaid
graph TD;
    A{Valor >= 3000?} -->|Sim| B{Periodo = Noturno?}
    A -->|Não| C[Normal]
    B -->|Sim| D[Fraude]
    B -->|Não| E[Normal]
```
