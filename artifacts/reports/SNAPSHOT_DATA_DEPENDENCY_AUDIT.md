# SNAPSHOT DATA DEPENDENCY AUDIT

### Data Sources
- `data/raw/orders.csv`: INTERNAL_TO_SNAPSHOT
- `data/raw/order_products__prior.csv`: INTERNAL_TO_SNAPSHOT 
- `data/processed/product_data.csv`: INTERNAL_TO_SNAPSHOT

### Undeclared External Dependencies
- NONE. Extensive python scanning confirmed that all paths resolve to `PROJECT_ROOT` internally.
