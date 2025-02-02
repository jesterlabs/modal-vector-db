import json
import uuid



def json_to_uuid(json_data, namespace=uuid.NAMESPACE_DNS):
    """Convert JSON data to a deterministic UUID."""
    json_str = json.dumps(json_data, sort_keys=True)  # Ensure consistent ordering
    return uuid.uuid5(namespace, json_str)

def build_where_clause(filters: dict) -> str:
    """
    Convert a dictionary of filters to a DuckDB WHERE clause.
    
    Supports operators:
    - eq: equals (default when no operator specified)
    - gt: greater than
    - gte: greater than or equal
    - lt: less than
    - lte: less than or equal
    - in: list membership
    - between: range inclusive
    
    Example:
    filters = {
        'type': 'fire',                    # equals
        'attack__gt': 80,                  # greater than
        'defense__lte': 100,               # less than or equal
        'name__in': ['charizard', 'moltres'], # in list
        'speed__between': [50, 100]        # between range
    }
    """
    operators = {
        'gt': '>',
        'gte': '>=',
        'lt': '<',
        'lte': '<=',
        'in': 'IN',
        'between': 'BETWEEN'
    }
    
    conditions = []
    
    for key, value in filters.items():
        parts = key.split('__')
        field = parts[0]
        op = parts[1] if len(parts) > 1 else 'eq'
        
        if op == 'eq':
            conditions.append(f"{field} = '{value}'" if isinstance(value, str) else f"{field} = {value}")
        elif op == 'in':
            formatted_values = [f"'{v}'" if isinstance(v, str) else str(v) for v in value]
            conditions.append(f"{field} IN ({', '.join(formatted_values)})")
        elif op == 'between':
            conditions.append(f"{field} BETWEEN {value[0]} AND {value[1]}")
        elif op in operators:
            value_str = f"'{value}'" if isinstance(value, str) else str(value)
            conditions.append(f"{field} {operators[op]} {value_str}")
            
    return ' AND '.join(conditions) if conditions else '1=1'