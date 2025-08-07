import React, { useState, useEffect, useRef } from 'react';
import './futuristic_theme.css';

// Toggle Component
export const Toggle = ({ 
    isActive = false, 
    onChange, 
    label, 
    disabled = false,
    size = 'default' 
}) => {
    const [active, setActive] = useState(isActive);
    
    const handleToggle = () => {
        if (!disabled) {
            const newState = !active;
            setActive(newState);
            onChange?.(newState);
        }
    };
    
    const sizeClasses = {
        small: 'w-12 h-6',
        default: 'w-14 h-8',
        large: 'w-16 h-10'
    };
    
    return (
        <div className="flex items-center gap-2">
            <div
                className={`toggle ${sizeClasses[size]} ${active ? 'active' : ''} ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
                onClick={handleToggle}
                role="switch"
                aria-checked={active}
                aria-label={label}
            >
                <div className="toggle-handle"></div>
            </div>
            {label && <span className="text-sm">{label}</span>}
        </div>
    );
};

// Dropdown Component
export const Dropdown = ({ 
    trigger, 
    children, 
    placement = 'bottom',
    width = 'auto'
}) => {
    const [isOpen, setIsOpen] = useState(false);
    const dropdownRef = useRef(null);
    
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsOpen(false);
            }
        };
        
        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);
    
    const placementClasses = {
        bottom: 'top-full left-0',
        bottomRight: 'top-full right-0',
        top: 'bottom-full left-0',
        topRight: 'bottom-full right-0'
    };
    
    return (
        <div className="dropdown" ref={dropdownRef}>
            <div 
                className="dropdown-trigger"
                onClick={() => setIsOpen(!isOpen)}
            >
                {trigger}
            </div>
            <div 
                className={`dropdown-menu ${isOpen ? 'active' : ''}`}
                style={{
                    [placement.includes('Right') ? 'right' : 'left']: '0',
                    width: width === 'auto' ? 'auto' : width
                }}
            >
                {children}
            </div>
        </div>
    );
};

// Dropdown Item Component
export const DropdownItem = ({ 
    children, 
    onClick, 
    icon,
    disabled = false 
}) => {
    return (
        <div
            className={`dropdown-item ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
            onClick={!disabled ? onClick : undefined}
        >
            {icon && <span className="mr-2">{icon}</span>}
            {children}
        </div>
    );
};

// Tooltip Component
export const Tooltip = ({ 
    children, 
    content, 
    placement = 'top',
    delay = 200 
}) => {
    const [isVisible, setIsVisible] = useState(false);
    const [timeoutId, setTimeoutId] = useState(null);
    
    const handleMouseEnter = () => {
        if (timeoutId) clearTimeout(timeoutId);
        setTimeoutId(setTimeout(() => setIsVisible(true), delay));
    };
    
    const handleMouseLeave = () => {
        if (timeoutId) clearTimeout(timeoutId);
        setIsVisible(false);
    };
    
    const placementClasses = {
        top: 'bottom-full left-1/2 transform -translate-x-1/2 mb-2',
        bottom: 'top-full left-1/2 transform -translate-x-1/2 mt-2',
        left: 'right-full top-1/2 transform -translate-y-1/2 mr-2',
        right: 'left-full top-1/2 transform -translate-y-1/2 ml-2'
    };
    
    return (
        <div 
            className="tooltip"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
        >
            {children}
            <div className={`tooltip-content ${placementClasses[placement]} ${isVisible ? 'visible' : ''}`}>
                {content}
            </div>
        </div>
    );
};

// Button Component
export const Button = ({ 
    children, 
    variant = 'default',
    size = 'default',
    icon,
    loading = false,
    disabled = false,
    onClick,
    className = '',
    ...props 
}) => {
    const variantClasses = {
        default: 'btn',
        primary: 'btn btn-primary',
        secondary: 'btn btn-secondary',
        ghost: 'btn btn-ghost'
    };
    
    const sizeClasses = {
        small: 'px-3 py-1 text-sm',
        default: 'px-4 py-2',
        large: 'px-6 py-3 text-lg'
    };
    
    return (
        <button
            className={`${variantClasses[variant]} ${sizeClasses[size]} ${className} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            onClick={!disabled && !loading ? onClick : undefined}
            disabled={disabled || loading}
            {...props}
        >
            {loading ? (
                <div className="loading mr-2"></div>
            ) : icon ? (
                <span className="mr-2">{icon}</span>
            ) : null}
            {children}
        </button>
    );
};

// Card Component
export const Card = ({ 
    children, 
    className = '',
    hover = true,
    onClick,
    ...props 
}) => {
    return (
        <div 
            className={`card ${hover ? 'card-hover' : ''} ${className}`}
            onClick={onClick}
            {...props}
        >
            {children}
        </div>
    );
};

// Input Component
export const Input = ({ 
    placeholder,
    value,
    onChange,
    type = 'text',
    disabled = false,
    error = false,
    className = '',
    ...props 
}) => {
    return (
        <input
            type={type}
            className={`input ${error ? 'input-error' : ''} ${className}`}
            placeholder={placeholder}
            value={value}
            onChange={onChange}
            disabled={disabled}
            {...props}
        />
    );
};

// Status Indicator Component
export const Status = ({ 
    type = 'default',
    children,
    className = ''
}) => {
    const typeClasses = {
        default: 'status',
        success: 'status status-success',
        warning: 'status status-warning',
        error: 'status status-error'
    };
    
    return (
        <div className={`${typeClasses[type]} ${className}`}>
            <div className="status-dot"></div>
            {children}
        </div>
    );
};

// Modal Component
export const Modal = ({ 
    isOpen, 
    onClose, 
    title, 
    children,
    size = 'default'
}) => {
    const modalRef = useRef(null);
    
    useEffect(() => {
        const handleEscape = (e) => {
            if (e.key === 'Escape') onClose();
        };
        
        if (isOpen) {
            document.addEventListener('keydown', handleEscape);
            document.body.style.overflow = 'hidden';
        }
        
        return () => {
            document.removeEventListener('keydown', handleEscape);
            document.body.style.overflow = 'unset';
        };
    }, [isOpen, onClose]);
    
    const sizeClasses = {
        small: 'max-w-md',
        default: 'max-w-lg',
        large: 'max-w-2xl',
        xlarge: 'max-w-4xl'
    };
    
    if (!isOpen) return null;
    
    return (
        <div className="modal-overlay" onClick={onClose}>
            <div 
                className={`modal ${sizeClasses[size]}`}
                onClick={(e) => e.stopPropagation()}
                ref={modalRef}
            >
                <div className="modal-header">
                    <h3 className="modal-title">{title}</h3>
                    <button className="modal-close" onClick={onClose}>
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 6L6 18M6 6l12 12"/>
                        </svg>
                    </button>
                </div>
                <div className="modal-content">
                    {children}
                </div>
            </div>
        </div>
    );
};

// Command Palette Component
export const CommandPalette = ({ 
    isOpen, 
    onClose, 
    commands = [],
    onExecute 
}) => {
    const [query, setQuery] = useState('');
    const [selectedIndex, setSelectedIndex] = useState(0);
    const inputRef = useRef(null);
    
    const filteredCommands = commands.filter(cmd => 
        cmd.name.toLowerCase().includes(query.toLowerCase()) ||
        cmd.description.toLowerCase().includes(query.toLowerCase())
    );
    
    useEffect(() => {
        if (isOpen && inputRef.current) {
            inputRef.current.focus();
        }
    }, [isOpen]);
    
    useEffect(() => {
        setSelectedIndex(0);
    }, [query]);
    
    const handleKeyDown = (e) => {
        switch (e.key) {
            case 'ArrowDown':
                e.preventDefault();
                setSelectedIndex(prev => 
                    prev < filteredCommands.length - 1 ? prev + 1 : 0
                );
                break;
            case 'ArrowUp':
                e.preventDefault();
                setSelectedIndex(prev => 
                    prev > 0 ? prev - 1 : filteredCommands.length - 1
                );
                break;
            case 'Enter':
                e.preventDefault();
                if (filteredCommands[selectedIndex]) {
                    onExecute?.(filteredCommands[selectedIndex]);
                    onClose();
                }
                break;
            case 'Escape':
                onClose();
                break;
        }
    };
    
    if (!isOpen) return null;
    
    return (
        <div className="command-palette-overlay" onClick={onClose}>
            <div className="command-palette" onClick={(e) => e.stopPropagation()}>
                <div className="command-palette-header">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="m21 21-4.35-4.35"/>
                    </svg>
                    <input
                        ref={inputRef}
                        type="text"
                        placeholder="Search commands..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        className="command-palette-input"
                    />
                </div>
                <div className="command-palette-list">
                    {filteredCommands.map((command, index) => (
                        <div
                            key={command.id}
                            className={`command-item ${index === selectedIndex ? 'selected' : ''}`}
                            onClick={() => {
                                onExecute?.(command);
                                onClose();
                            }}
                        >
                            <div className="command-icon">{command.icon}</div>
                            <div className="command-content">
                                <div className="command-name">{command.name}</div>
                                <div className="command-description">{command.description}</div>
                            </div>
                            <div className="command-shortcut">{command.shortcut}</div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

// Main App Component
export const SOPHIEInterface = () => {
    const [commandPaletteOpen, setCommandPaletteOpen] = useState(false);
    const [systemStatus, setSystemStatus] = useState({
        unifiedExecution: true,
        securityScaffold: false,
        reflexiveMoE: true
    });
    
    const commands = [
        {
            id: 'execute',
            name: 'Execute Command',
            description: 'Run a command through SOPHIE',
            icon: '⚡',
            shortcut: 'Ctrl+E'
        },
        {
            id: 'logs',
            name: 'View Logs',
            description: 'Check system logs and audit trail',
            icon: '📋',
            shortcut: 'Ctrl+L'
        },
        {
            id: 'security',
            name: 'Security Settings',
            description: 'Manage vault and OAuth settings',
            icon: '🔐',
            shortcut: 'Ctrl+S'
        },
        {
            id: 'settings',
            name: 'Preferences',
            description: 'Configure SOPHIE settings',
            icon: '⚙️',
            shortcut: 'Ctrl+,'
        }
    ];
    
    useEffect(() => {
        const handleKeyDown = (e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                setCommandPaletteOpen(true);
            }
        };
        
        document.addEventListener('keydown', handleKeyDown);
        return () => document.removeEventListener('keydown', handleKeyDown);
    }, []);
    
    const handleToggle = (key) => (value) => {
        setSystemStatus(prev => ({ ...prev, [key]: value }));
    };
    
    const handleCommandExecute = (command) => {
        console.log('Executing command:', command.name);
        // Implement command execution logic
    };
    
    return (
        <div className="sophie-interface">
            {/* Header */}
            <header className="header">
                <div className="container">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                            <div className="sophie-logo">
                                <span>S</span>
                            </div>
                            <h1 className="text-xl font-semibold">SOPHIE</h1>
                        </div>
                        
                        <nav className="flex items-center gap-4">
                            <Dropdown
                                trigger={
                                    <div className="flex items-center gap-2">
                                        <span>Settings</span>
                                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                            <path d="m6 9 6 6 6-6"/>
                                        </svg>
                                    </div>
                                }
                            >
                                <DropdownItem icon="👤">Profile</DropdownItem>
                                <DropdownItem icon="⚙️">Preferences</DropdownItem>
                                <DropdownItem icon="🔐">Security</DropdownItem>
                                <DropdownItem icon="❓">Help</DropdownItem>
                            </Dropdown>
                            
                            <Tooltip content="Get help and support">
                                <Button variant="ghost" icon="❓" />
                            </Tooltip>
                        </nav>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="main">
                <div className="container">
                    {/* Hero Section */}
                    <div className="text-center mb-4">
                        <h1 className="fade-in">Welcome to SOPHIE</h1>
                        <p className="text-secondary fade-in">
                            The future is quiet, responsive, and already understands you.
                        </p>
                    </div>

                    {/* Control Panel */}
                    <div className="grid grid-2 mb-4">
                        <Card className="slide-up">
                            <h3>System Status</h3>
                            <div className="flex items-center gap-2 mb-2">
                                <Status type="success">Online</Status>
                            </div>
                            <p className="text-sm text-secondary">All systems operational</p>
                        </Card>

                        <Card className="slide-up">
                            <h3>Active Sessions</h3>
                            <div className="flex items-center gap-2 mb-2">
                                <span className="text-2xl font-semibold">3</span>
                                <span className="text-secondary">active connections</span>
                            </div>
                            <p className="text-sm text-secondary">2 authenticated, 1 pending</p>
                        </Card>
                    </div>

                    {/* Features Grid */}
                    <div className="grid grid-3 mb-4">
                        <Card>
                            <h4>Unified Execution</h4>
                            <p className="text-sm text-secondary mb-3">
                                Execute commands across any tool, API, or environment with trust-based approval.
                            </p>
                            <Toggle
                                isActive={systemStatus.unifiedExecution}
                                onChange={handleToggle('unifiedExecution')}
                                label={systemStatus.unifiedExecution ? 'Enabled' : 'Disabled'}
                            />
                        </Card>

                        <Card>
                            <h4>Security Scaffold</h4>
                            <p className="text-sm text-secondary mb-3">
                                Vault-based secret management with OAuth support and audit logging.
                            </p>
                            <Toggle
                                isActive={systemStatus.securityScaffold}
                                onChange={handleToggle('securityScaffold')}
                                label={systemStatus.securityScaffold ? 'Enabled' : 'Disabled'}
                            />
                        </Card>

                        <Card>
                            <h4>Reflexive MoE</h4>
                            <p className="text-sm text-secondary mb-3">
                                Multi-expert orchestration with dynamic role selection and trust scoring.
                            </p>
                            <Toggle
                                isActive={systemStatus.reflexiveMoE}
                                onChange={handleToggle('reflexiveMoE')}
                                label={systemStatus.reflexiveMoE ? 'Enabled' : 'Disabled'}
                            />
                        </Card>
                    </div>

                    {/* Action Panel */}
                    <Card>
                        <h3>Quick Actions</h3>
                        <div className="grid grid-4">
                            <Button variant="primary" icon="⚡">
                                Execute Command
                            </Button>
                            
                            <Button icon="📋">
                                View Logs
                            </Button>
                            
                            <Button icon="🔐">
                                Security
                            </Button>
                            
                            <Button icon="⚙️">
                                Settings
                            </Button>
                        </div>
                    </Card>
                </div>
            </main>

            {/* Command Palette */}
            <CommandPalette
                isOpen={commandPaletteOpen}
                onClose={() => setCommandPaletteOpen(false)}
                commands={commands}
                onExecute={handleCommandExecute}
            />
        </div>
    );
};

export default SOPHIEInterface; 