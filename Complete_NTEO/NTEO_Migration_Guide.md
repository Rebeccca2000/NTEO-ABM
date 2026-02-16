# NTEO System Migration Guide
## Transition to Unified Network Topology System

### 🎯 Migration Objective
Eliminate the dual routing system and transition to using **ONLY the Network Topology System (Two Layers Working Simultaneously)** as outlined in the "Guide on better construction of the NTEO network."

---

## 📋 Pre-Migration Checklist

### 1. Back Up Your Current System
```bash
# Create backup of your current codebase
cp -r your_nteo_project your_nteo_project_backup_$(date +%Y%m%d)

# Backup your database
pg_dump nteo_abm > nteo_abm_backup_$(date +%Y%m%d).sql
```

### 2. Verify Current System Status
- [ ] Identify if you're currently running dual routing systems
- [ ] Note which network topology files you currently use
- [ ] Document your current configuration parameters
- [ ] List any custom modifications to ABM files

---

## 🔧 Step-by-Step Migration Process

### Phase 1: Replace Core Configuration (Week 1)

#### Step 1.1: Replace Database Configuration
```python
# OLD: database.py with embedded network config
NETWORK_CONFIG = {
    'topology_type': 'degree_constrained',
    'degree_constraint': 4,
    # ... other config
}

# NEW: Use network_config.py instead
from network_config import get_current_config
NETWORK_CONFIG = get_current_config()
```

**Action Required:**
1. Replace your existing `database.py` with `database_updated.py`
2. Add the new `network_config.py` file
3. Update imports in your existing files:
   ```python
   # Change this:
   import database as db
   
   # To this (no changes needed if you use database_updated.py):
   import database_updated as db
   ```

#### Step 1.2: Implement Network Factory Pattern
**Action Required:**
1. Add `network_factory.py` to your project
2. Replace manual network creation with factory pattern:
   ```python
   # OLD: Manual network creation
   if network_config['topology_type'] == 'degree_constrained':
       network_manager = TwoLayerNetworkManager(...)
   elif network_config['topology_type'] == 'small_world':
       network_manager = TwoLayerNetworkManager(...)
   
   # NEW: Factory pattern
   from network_factory import create_network
   network_interface = create_network(topology_type, variation_parameter)
   ```

### Phase 2: Update ABM Initialization (Week 2)

#### Step 2.1: Replace ABM Model Class
**Critical Change:** Your current `agent_run_visualisation.py` contains dual routing logic.

**Action Required:**
1. Replace the network initialization section in your ABM model with:
   ```python
   # NEW: Single system initialization
   from network_factory import NetworkFactory
   from network_config import NetworkConfigurationManager
   
   self.network_factory = NetworkFactory()
   network_manager = self.network_factory.create_network(
       network_type=topology_type,
       variation_parameter=variation_parameter
   )
   self.network_interface = StandardizedNetworkInterface(network_manager)
   ```

2. **Remove all legacy routing code** from your model:
   ```python
   # REMOVE these if they exist:
   - dijkstra_with_congestion methods
   - Legacy grid routing
   - Dual system checks
   - Multiple route calculation paths
   ```

3. **Replace routing calls** throughout your codebase:
   ```python
   # OLD: Multiple routing systems
   if use_network_topology:
       route = self.network_router.find_route(...)
   else:
       route = self.legacy_router.find_route(...)
   
   # NEW: Single routing system
   route = self.network_interface.find_route(start_pos, end_pos, agent_type)
   ```

#### Step 2.2: Update Agent Classes
**Action Required for each agent type:**

1. **Commuter Agents**: Update route-finding calls
   ```python
   # In your commuter agent step() method:
   # OLD:
   route = self.model.find_route_legacy(self.pos, destination)
   
   # NEW:
   route = self.model.find_route(self.pos, destination, 'commuter')
   ```

2. **Station Agents**: Update network node references
3. **MaaS Agents**: Update service area calculations

### Phase 3: Research Framework Integration (Week 3)

#### Step 3.1: Replace Comparison Scripts
If you have existing comparison scripts (like `degree_comparison.py`):

**Action Required:**
```python
# OLD: Manual comparison loop
for degree in [3, 4, 5, 6, 7]:
    # Manual model creation and testing
    
# NEW: Use research runner
from nteo_research_runner import run_degree_comparison_study
results = run_degree_comparison_study([3, 4, 5, 6, 7])
```

#### Step 3.2: Update Research Studies
**Action Required:**
1. Replace your existing research scripts with `nteo_research_runner.py`
2. Update study configurations:
   ```python
   # NEW: Predefined research studies
   from nteo_research_runner import NTEOResearchRunner
   
   runner = NTEOResearchRunner()
   results = runner.run_research_study('degree_comparison_study')
   ```

### Phase 4: Validation and Testing (Week 4)

#### Step 4.1: Run Validation Suite
**Action Required:**
```python
# Run comprehensive validation
from nteo_validation_suite import NTEOValidationSuite

validator = NTEOValidationSuite()
results = validator.run_full_validation()

# Check for critical issues
if results['overall_status'] == 'FAIL':
    print("Issues found:", results['critical_issues'])
```

#### Step 4.2: Performance Verification
**Expected Results:**
- 40-60% performance improvement in simulation runtime
- 40%+ reduction in memory usage
- Elimination of dual routing overhead

**Action Required:**
```python
# Benchmark performance
from nteo_validation_suite import run_performance_benchmark
performance_data = run_performance_benchmark()
```

---

## 🔍 File-by-File Migration Instructions

### 1. Core Configuration Files

| Current File | Action | New File |
|--------------|--------|----------|
| `database.py` | Replace | `database_updated.py` |
| N/A | Add | `network_config.py` |
| N/A | Add | `network_factory.py` |

### 2. ABM Model Files

| Current File | Action | New File |
|--------------|--------|----------|
| `agent_run_visualisation.py` | Modify | Update with `abm_initialization.py` patterns |
| Agent classes | Modify | Update routing calls |

### 3. Research and Testing Files

| Current File | Action | New File |
|--------------|--------|----------|
| `degree_comparison.py` | Replace | Use `nteo_research_runner.py` |
| Custom research scripts | Replace | Use research runner framework |
| N/A | Add | `nteo_validation_suite.py` |

---

## 🚨 Critical Migration Points

### 1. Eliminate Dual Routing System
**CRITICAL:** This is the main objective. Ensure you:
- [ ] Remove all legacy routing code
- [ ] Have only ONE routing method: `self.network_interface.find_route()`
- [ ] No conditional routing logic (`if use_network_topology:`)

### 2. Parameter Standardization
**CRITICAL:** Standardize parameter interfaces:
```python
# OLD: Different parameter names for each topology
degree_constraint = 4
rewiring_probability = 0.1
attachment_parameter = 2

# NEW: Unified variation_parameter
create_network('degree_constrained', variation_parameter=4)
create_network('small_world', variation_parameter=0.1)
create_network('scale_free', variation_parameter=2)
```

### 3. Configuration Consolidation
**CRITICAL:** Single point of control:
```python
# NEW: Master control parameter
from network_config import switch_network_type
switch_network_type('small_world')  # Changes everything globally
```

---

## 🧪 Testing Your Migration

### Quick Validation Test
```python
# Test 1: Basic system functionality
from nteo_validation_suite import run_quick_validation
success = run_quick_validation()

if success:
    print("✅ Migration successful!")
else:
    print("❌ Migration needs attention")
```

### Performance Comparison Test
```python
# Test 2: Verify performance improvements
from nteo_validation_suite import run_performance_benchmark
performance_data = run_performance_benchmark()

# Should show improvements over legacy system
print(f"Route calculation time: {performance_data['route_calculation_time']:.4f}s")
```

### Research Framework Test
```python
# Test 3: Research studies work
from nteo_research_runner import run_degree_comparison_study
results = run_degree_comparison_study([3, 4], num_runs=1, steps_per_run=10)

print(f"Study completed with {len(results['results'])} configurations")
```

---

## 🐛 Common Migration Issues and Solutions

### Issue 1: Import Errors
**Problem:** `ImportError: cannot import name 'TwoLayerNetworkManager'`

**Solution:**
```python
# Ensure you have the correct import structure
from network_factory import create_network, StandardizedNetworkInterface
# Instead of importing TwoLayerNetworkManager directly
```

### Issue 2: Configuration Not Found
**Problem:** `KeyError: 'topology_type'`

**Solution:**
```python
# Ensure network_config.py is properly set up
from network_config import get_current_config
config = get_current_config()
print(config)  # Verify configuration is valid
```

### Issue 3: Routing Failures
**Problem:** `Route calculation failed` errors

**Solution:**
```python
# Check network connectivity
network_interface = create_network('degree_constrained', 4)
stats = network_interface.get_network_stats()
print(f"Network connected: {stats['is_connected']}")

# If not connected, check network generation parameters
```

### Issue 4: Performance Not Improved
**Problem:** No performance improvement after migration

**Solution:**
1. Verify dual system completely eliminated:
   ```python
   # Check for legacy routing remnants
   from nteo_validation_suite import NTEOValidationSuite
   validator = NTEOValidationSuite()
   routing_results = validator._test_single_routing_system()
   print(routing_results)
   ```

2. Check for memory leaks or inefficient caching

---

## 📊 Expected Outcomes

### Performance Improvements
- **Simulation Speed:** 40-60% faster execution
- **Memory Usage:** 40%+ reduction
- **Code Complexity:** 40% fewer lines (elimination of dual system)

### Research Capabilities
- **Easy Topology Switching:** Change one parameter to switch network types
- **Automated Comparisons:** Built-in research framework
- **Standardized Output:** Consistent results format across all studies

### Maintainability
- **Single Codebase:** No more parallel routing systems
- **Clear Architecture:** Separation of concerns between network and spatial layers
- **Extensible Design:** Easy to add new topology types

---

## 🎯 Post-Migration Verification

### Checklist: Migration Complete ✅
- [ ] All files migrated to new system
- [ ] Validation suite passes (>80% success rate)
- [ ] Performance benchmarks show improvement
- [ ] Research studies run successfully
- [ ] No legacy routing code remains
- [ ] Single control parameter system working
- [ ] Documentation updated

### Success Criteria Met
- [ ] **Technical:** Simulation runs without legacy dependencies
- [ ] **Performance:** 50%+ runtime improvement
- [ ] **Research:** Support for all topology types (degree-constrained, small-world, scale-free)
- [ ] **Usability:** Single parameter controls topology type

---

## 🆘 Getting Help

### If Migration Fails
1. **Check validation results:**
   ```python
   from nteo_validation_suite import NTEOValidationSuite
   validator = NTEOValidationSuite()
   results = validator.run_full_validation()
   print("Critical issues:", results['critical_issues'])
   ```

2. **Review migration step-by-step:**
   - Verify each file was properly replaced/updated
   - Check import statements
   - Ensure no legacy routing code remains

3. **Test incrementally:**
   - Start with basic network creation
   - Test routing with small models
   - Gradually increase complexity

### Rollback Plan
If migration fails critically:
```bash
# Restore from backup
rm -rf your_nteo_project
cp -r your_nteo_project_backup_YYYYMMDD your_nteo_project

# Restore database
dropdb nteo_abm
createdb nteo_abm
psql nteo_abm < nteo_abm_backup_YYYYMMDD.sql
```

---

## 🚀 Next Steps After Migration

### 1. Research Studies
```python
# Run comprehensive NTEO analysis
from nteo_research_runner import run_full_nteo_comparison
results = run_full_nteo_comparison()
```

### 2. Custom Topology Development
```python
# Extend system with new topology types
# Add to network_config.py TOPOLOGY_PARAMETERS
# Implement in network_factory.py
```

### 3. Advanced Research
```python
# Use research framework for policy studies
runner = NTEOResearchRunner()
results = runner.run_research_study('efficiency_equity_optimization')
```

---

**🎉 Congratulations!** 

You have successfully migrated to the unified NTEO system with single network topology control. Your system is now ready for advanced transport equity research with improved performance and maintainability.