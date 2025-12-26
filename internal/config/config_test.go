package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSpecTypesAndGroups(t *testing.T) {
	// テスト用の設定ファイルを作成
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, ".specverify.yml")

	configContent := `
specs_dir: specs/
code_dir: src/
ai_provider: claude

spec_types:
  ui:
    code_paths:
      - client/components
      - client/pages
    verification_focus:
      - コンポーネント構成
      - 画面遷移

  api:
    code_paths:
      - server/routes
    verification_focus:
      - エンドポイント定義
      - リクエスト/レスポンス形式

  domain:
    code_paths:
      - server/domain
    verification_focus:
      - ビジネスルール
      - ドメインロジック

groups:
  frontend:
    types: [ui]
    description: "フロントエンド関連"
  backend:
    types: [api, domain]
    description: "バックエンド関連"
  all:
    types: [ui, api, domain]
    description: "全て"

options:
  concurrency: 3
  pass_threshold: 50
`
	if err := os.WriteFile(configFile, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	// 設定を読み込む
	cfg, err := Load(configFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// spec_types のテスト
	t.Run("GetCodePaths", func(t *testing.T) {
		paths := cfg.GetCodePaths("ui")
		if len(paths) != 2 {
			t.Errorf("Expected 2 paths for ui, got %d", len(paths))
		}
		if paths[0] != "src/client/components" {
			t.Errorf("Expected 'src/client/components', got '%s'", paths[0])
		}
	})

	t.Run("GetVerificationFocus", func(t *testing.T) {
		focus := cfg.GetVerificationFocus("api")
		if len(focus) != 2 {
			t.Errorf("Expected 2 focus items for api, got %d", len(focus))
		}
		if focus[0] != "エンドポイント定義" {
			t.Errorf("Expected 'エンドポイント定義', got '%s'", focus[0])
		}
	})

	t.Run("GetAllSpecTypes", func(t *testing.T) {
		types := cfg.GetAllSpecTypes()
		if len(types) != 3 {
			t.Errorf("Expected 3 types, got %d", len(types))
		}
	})

	t.Run("HasSpecType", func(t *testing.T) {
		if !cfg.HasSpecType("domain") {
			t.Error("Expected domain to exist")
		}
		if cfg.HasSpecType("nonexistent") {
			t.Error("Expected nonexistent to not exist")
		}
	})

	// groups のテスト
	t.Run("GetTypesByGroup", func(t *testing.T) {
		types := cfg.GetTypesByGroup("backend")
		if len(types) != 2 {
			t.Errorf("Expected 2 types in backend group, got %d", len(types))
		}
	})

	t.Run("GetAllGroups", func(t *testing.T) {
		groups := cfg.GetAllGroups()
		if len(groups) != 3 {
			t.Errorf("Expected 3 groups, got %d", len(groups))
		}
	})

	t.Run("HasGroup", func(t *testing.T) {
		if !cfg.HasGroup("frontend") {
			t.Error("Expected frontend group to exist")
		}
		if cfg.HasGroup("nonexistent") {
			t.Error("Expected nonexistent group to not exist")
		}
	})

	t.Run("GetSpecTypeInfo", func(t *testing.T) {
		info := cfg.GetSpecTypeInfo("domain")
		if info == nil {
			t.Fatal("Expected domain info to exist")
		}
		if len(info.CodePaths) != 1 {
			t.Errorf("Expected 1 code path, got %d", len(info.CodePaths))
		}
		if len(info.VerificationFocus) != 2 {
			t.Errorf("Expected 2 verification focus items, got %d", len(info.VerificationFocus))
		}
	})
}

func TestBackwardCompatibility(t *testing.T) {
	// 従来形式の設定ファイルを作成
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, ".specverify.yml")

	configContent := `
specs_dir: specs/
code_dir: src/
ai_provider: claude

mapping:
  ui: client/components
  api: server/routes

options:
  concurrency: 3
  pass_threshold: 50
`
	if err := os.WriteFile(configFile, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	cfg, err := Load(configFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	// mapping から GetCodePath が動作することを確認
	t.Run("GetCodePath from mapping", func(t *testing.T) {
		path := cfg.GetCodePath("ui")
		if path != "src/client/components" {
			t.Errorf("Expected 'src/client/components', got '%s'", path)
		}
	})

	// mapping から GetCodePaths が動作することを確認
	t.Run("GetCodePaths from mapping", func(t *testing.T) {
		paths := cfg.GetCodePaths("api")
		if len(paths) != 1 {
			t.Errorf("Expected 1 path, got %d", len(paths))
		}
		if paths[0] != "src/server/routes" {
			t.Errorf("Expected 'src/server/routes', got '%s'", paths[0])
		}
	})

	// mapping から GetAllSpecTypes が動作することを確認
	t.Run("GetAllSpecTypes from mapping", func(t *testing.T) {
		types := cfg.GetAllSpecTypes()
		if len(types) != 2 {
			t.Errorf("Expected 2 types, got %d", len(types))
		}
	})

	// mapping から HasSpecType が動作することを確認
	t.Run("HasSpecType from mapping", func(t *testing.T) {
		if !cfg.HasSpecType("ui") {
			t.Error("Expected ui to exist from mapping")
		}
	})

	// mapping から GetSpecTypeInfo が動作することを確認
	t.Run("GetSpecTypeInfo from mapping", func(t *testing.T) {
		info := cfg.GetSpecTypeInfo("api")
		if info == nil {
			t.Fatal("Expected api info to exist from mapping")
		}
		if len(info.CodePaths) != 1 {
			t.Errorf("Expected 1 code path, got %d", len(info.CodePaths))
		}
		if info.CodePaths[0] != "server/routes" {
			t.Errorf("Expected 'server/routes', got '%s'", info.CodePaths[0])
		}
	})
}

func TestLoadOptions(t *testing.T) {
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, ".specverify.yml")

	configContent := `
specs_dir: specs/
code_dir: src/
ai_provider: claude
ai_api_key: config-key
`
	if err := os.WriteFile(configFile, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	// 環境変数をクリア
	savedKeys := map[string]string{
		"SPEC_VERIFY_API_KEY": os.Getenv("SPEC_VERIFY_API_KEY"),
		"ANTHROPIC_API_KEY":   os.Getenv("ANTHROPIC_API_KEY"),
		"OPENAI_API_KEY":      os.Getenv("OPENAI_API_KEY"),
	}
	defer func() {
		for k, v := range savedKeys {
			if v != "" {
				os.Setenv(k, v)
			} else {
				os.Unsetenv(k)
			}
		}
	}()
	for k := range savedKeys {
		os.Unsetenv(k)
	}

	t.Run("WithAPIKey overrides config file", func(t *testing.T) {
		cfg, err := Load(configFile, WithAPIKey("cli-key"))
		if err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		if cfg.AIAPIKey != "cli-key" {
			t.Errorf("Expected 'cli-key', got '%s'", cfg.AIAPIKey)
		}
	})

	t.Run("WithProvider changes provider", func(t *testing.T) {
		cfg, err := Load(configFile, WithProvider("openai"))
		if err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		if cfg.AIProvider != "openai" {
			t.Errorf("Expected 'openai', got '%s'", cfg.AIProvider)
		}
	})

	t.Run("WithProvider affects API key lookup", func(t *testing.T) {
		// OpenAI用の環境変数をセット
		os.Setenv("OPENAI_API_KEY", "openai-env-key")
		defer os.Unsetenv("OPENAI_API_KEY")

		cfg, err := Load(configFile, WithProvider("openai"))
		if err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		if cfg.AIAPIKey != "openai-env-key" {
			t.Errorf("Expected 'openai-env-key', got '%s'", cfg.AIAPIKey)
		}
	})

	t.Run("CLI API key has highest priority", func(t *testing.T) {
		os.Setenv("OPENAI_API_KEY", "env-key")
		defer os.Unsetenv("OPENAI_API_KEY")

		cfg, err := Load(configFile, WithProvider("openai"), WithAPIKey("cli-key"))
		if err != nil {
			t.Fatalf("Load failed: %v", err)
		}
		if cfg.AIAPIKey != "cli-key" {
			t.Errorf("Expected CLI key 'cli-key', got '%s'", cfg.AIAPIKey)
		}
	})
}

func TestHasAPIKeyOption(t *testing.T) {
	t.Run("returns true when WithAPIKey is used", func(t *testing.T) {
		opts := []LoadOption{WithAPIKey("test-key")}
		if !hasAPIKeyOption(opts) {
			t.Error("Expected true, got false")
		}
	})

	t.Run("returns false when WithAPIKey is empty", func(t *testing.T) {
		opts := []LoadOption{WithAPIKey("")}
		if hasAPIKeyOption(opts) {
			t.Error("Expected false, got true")
		}
	})

	t.Run("returns false when only WithProvider is used", func(t *testing.T) {
		opts := []LoadOption{WithProvider("openai")}
		if hasAPIKeyOption(opts) {
			t.Error("Expected false, got true")
		}
	})

	t.Run("returns false when no options", func(t *testing.T) {
		opts := []LoadOption{}
		if hasAPIKeyOption(opts) {
			t.Error("Expected false, got true")
		}
	})
}

func TestSpecTypesPriorityOverMapping(t *testing.T) {
	// spec_types と mapping 両方が定義された場合、spec_types が優先されることを確認
	tmpDir := t.TempDir()
	configFile := filepath.Join(tmpDir, ".specverify.yml")

	configContent := `
specs_dir: specs/
code_dir: src/
ai_provider: claude

mapping:
  ui: old/path

spec_types:
  ui:
    code_paths:
      - new/path
    verification_focus:
      - 新しい検証観点

options:
  concurrency: 3
`
	if err := os.WriteFile(configFile, []byte(configContent), 0644); err != nil {
		t.Fatalf("Failed to write config file: %v", err)
	}

	cfg, err := Load(configFile)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	t.Run("spec_types has priority over mapping", func(t *testing.T) {
		path := cfg.GetCodePath("ui")
		if path != "src/new/path" {
			t.Errorf("Expected spec_types path 'src/new/path', got '%s'", path)
		}
	})
}
