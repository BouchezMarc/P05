import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, call
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    query_init_df, 
    insert_train_data, 
    execute_sql_file,
    table_is_empty,
    create_bd_base,
    insert_data_into_db
)
from sqlalchemy.exc import SQLAlchemyError


class TestUtils:
    """Tests pour les utilitaires"""
    
    # =========== Tests pour execute_sql_file ===========
    @patch("builtins.open", new_callable=mock_open, read_data="SELECT * FROM test;")
    def test_execute_sql_file_success(self, mock_file):
        """Test l'exécution réussie d'un fichier SQL"""
        mock_conn = MagicMock()
        
        execute_sql_file(mock_conn, "/fake/path.sql")
        
        # Vérifier que le fichier a été ouvert
        mock_file.assert_called_once_with("/fake/path.sql", "r", encoding="utf-8")
        
        # Vérifier que execute et commit ont été appelés
        assert mock_conn.execute.called
        mock_conn.commit.assert_called_once()
    
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_execute_sql_file_file_not_found(self, mock_file):
        """Test avec un fichier SQL inexistant"""
        mock_conn = MagicMock()
        
        with pytest.raises(FileNotFoundError):
            execute_sql_file(mock_conn, "/fake/nonexistent.sql")
    
    # =========== Tests pour table_is_empty ===========
    def test_table_is_empty_true(self):
        """Test quand la table est vide"""
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (0,)
        mock_db.execute.return_value = mock_result
        
        result = table_is_empty(mock_db, "test_table")
        
        assert result is True
        mock_db.execute.assert_called_once()
    
    def test_table_is_empty_false(self):
        """Test quand la table contient des données"""
        mock_db = MagicMock()
        mock_result = MagicMock()
        mock_result.fetchone.return_value = (5,)
        mock_db.execute.return_value = mock_result
        
        result = table_is_empty(mock_db, "test_table")
        
        assert result is False
    
    # =========== Tests pour query_init_df ===========
    @patch("src.utils.SessionLocal")
    def test_query_init_df_returns_dataframe(self, mock_session):
        """Test que query_init_df retourne un DataFrame avec des données"""
        # Mock la session
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        # Mock des objets ViewRh
        mock_row1 = MagicMock()
        mock_row1.__dict__ = {
            '_sa_instance_state': 'dummy',
            'id_employee': 1,
            'age': 30,
            'a_quitte_l_entreprise': 'Non'
        }
        mock_row2 = MagicMock()
        mock_row2.__dict__ = {
            '_sa_instance_state': 'dummy',
            'id_employee': 2,
            'age': 40,
            'a_quitte_l_entreprise': 'Oui'
        }
        
        mock_db.query.return_value.all.return_value = [mock_row1, mock_row2]
        
        result = query_init_df()
        
        # Vérifier que c'est un DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert 'id_employee' in result.columns
        assert '_sa_instance_state' not in result.columns
        mock_db.close.assert_called()
    
    @patch("src.utils.SessionLocal")
    def test_query_init_df_empty_data(self, mock_session):
        """Test query_init_df avec une table vide"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query.return_value.all.return_value = []
        
        with pytest.raises(ValueError, match="Aucune donnée dans la BDD"):
            query_init_df()
        
        mock_db.close.assert_called()
    
    @patch("src.utils.SessionLocal")
    def test_query_init_df_exception(self, mock_session):
        """Test query_init_df avec une exception"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.query.side_effect = Exception("Database error")
        
        with pytest.raises(Exception):
            query_init_df()
        
        mock_db.close.assert_called()
    
    # =========== Tests pour insert_data_into_db ===========
    @patch("src.utils.TInputs")
    def test_insert_data_into_db_success(self, mock_tinputs_class):
        """Test l'insertion de données dans la DB"""
        mock_db = MagicMock()
        
        # Mock des colonnes
        mock_column1 = MagicMock()
        mock_column1.name = 'id_employee'
        mock_column1.nullable = False
        mock_column1.default = None
        mock_column1.server_default = None
        
        mock_column2 = MagicMock()
        mock_column2.name = 'age'
        mock_column2.nullable = False
        mock_column2.default = None
        mock_column2.server_default = None
        
        mock_column3 = MagicMock()
        mock_column3.name = 'id_input'
        mock_column3.nullable = True
        
        # Mock de columns qui supporte keys() et l'itération
        mock_columns = MagicMock()
        mock_columns.keys.return_value = ['id_employee', 'age', 'id_input']
        mock_columns.__iter__.return_value = iter([mock_column1, mock_column2, mock_column3])
        
        # Mock de __table__
        mock_table = MagicMock()
        mock_table.columns = mock_columns
        mock_tinputs_class.__table__ = mock_table
        
        # DataFrame de test
        df = pd.DataFrame({
            'id_employee': [100, 200],
            'age': [30, 40]
        })
        
        insert_data_into_db(df, mock_db)
        
        # Vérifier que add_all et flush ont été appelés
        mock_db.add_all.assert_called_once()
        mock_db.flush.assert_called_once()
    
    @patch("src.utils.TInputs")
    def test_insert_data_into_db_missing_columns(self, mock_tinputs_class):
        """Test insert_data_into_db avec des colonnes manquantes"""
        mock_db = MagicMock()
        
        # Mock une colonne requise manquante
        mock_column = MagicMock()
        mock_column.name = 'required_col'
        mock_column.nullable = False
        mock_column.default = None
        mock_column.server_default = None
        
        mock_column2 = MagicMock()
        mock_column2.name = 'id_employee'
        mock_column2.nullable = True
        
        # Mock de columns qui supporte keys() et l'itération
        mock_columns = MagicMock()
        mock_columns.keys.return_value = ['id_employee', 'required_col']
        mock_columns.__iter__.return_value = iter([mock_column, mock_column2])
        
        # Mock de __table__
        mock_table = MagicMock()
        mock_table.columns = mock_columns
        mock_tinputs_class.__table__ = mock_table
        
        # DataFrame sans la colonne requise
        df = pd.DataFrame({
            'id_employee': [100]
        })
        
        with pytest.raises(ValueError, match="Colonnes requises manquantes"):
            insert_data_into_db(df, mock_db)
    
    @patch("src.utils.TInputs")
    def test_insert_data_into_db_exception(self, mock_tinputs_class):
        """Test insert_data_into_db avec exception lors du flush"""
        mock_db = MagicMock()
        
        mock_column = MagicMock()
        mock_column.name = 'id_employee'
        mock_column.nullable = False
        mock_column.default = None
        mock_column.server_default = None
        
        # Mock de columns qui supporte keys() et l'itération
        mock_columns = MagicMock()
        mock_columns.keys.return_value = ['id_employee']
        mock_columns.__iter__.return_value = iter([mock_column])
        
        # Mock de __table__
        mock_table = MagicMock()
        mock_table.columns = mock_columns
        mock_tinputs_class.__table__ = mock_table
        
        # Exception lors du flush
        mock_db.flush.side_effect = Exception("Flush error")
        
        df = pd.DataFrame({'id_employee': [100]})
        
        with pytest.raises(Exception, match="Flush error"):
            insert_data_into_db(df, mock_db)
        
        mock_db.add_all.assert_called_once()
        mock_db.rollback.assert_called_once()
    
    # =========== Tests pour insert_train_data ===========
    @patch("src.utils.SessionLocal")
    @patch("src.utils.insert_data_into_db")
    def test_insert_train_data_success(self, mock_insert_func, mock_session):
        """Test insertion de données d'entraînement"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        
        df = pd.DataFrame({
            "id_employee": [100, 200],
            "age": [30, 40]
        })
        
        insert_train_data(df)
        
        # Vérifier que insert_data_into_db a été appelé
        mock_insert_func.assert_called_once_with(df, mock_db)
        mock_db.commit.assert_called_once()
        mock_db.close.assert_called_once()
    
    @patch("src.utils.SessionLocal")
    @patch("src.utils.insert_data_into_db")
    def test_insert_train_data_exception(self, mock_insert_func, mock_session):
        """Test insert_train_data avec exception SQLAlchemy"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_insert_func.side_effect = SQLAlchemyError("Insert error")
        
        df = pd.DataFrame({"id_employee": [100]})
        
        insert_train_data(df)
        
        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()
    
    # =========== Tests pour create_bd_base ===========
    @patch("src.utils.SessionLocal")
    @patch("src.utils.execute_sql_file")
    @patch("src.utils.table_is_empty")
    def test_create_bd_base_success_empty_tables(self, mock_is_empty, mock_exec_sql, mock_session):
        """Test création de la base avec tables vides"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.autocommit = False
        
        # Toutes les tables sont vides
        mock_is_empty.return_value = True
        
        create_bd_base()
        
        # Vérifier que execute_sql_file a été appelé 4 fois (1 create + 3 inserts)
        assert mock_exec_sql.call_count == 4
        mock_db.close.assert_called_once()
    
    @patch("src.utils.SessionLocal")
    @patch("src.utils.execute_sql_file")
    @patch("src.utils.table_is_empty")
    def test_create_bd_base_tables_already_filled(self, mock_is_empty, mock_exec_sql, mock_session):
        """Test création de la base avec tables déjà remplies"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_db.autocommit = False
        
        # Toutes les tables sont déjà remplies
        mock_is_empty.return_value = False
        
        create_bd_base()
        
        # Vérifier que execute_sql_file a été appelé 1 fois (seulement create)
        assert mock_exec_sql.call_count == 1
        mock_db.close.assert_called_once()
    
    @patch("src.utils.SessionLocal")
    @patch("src.utils.execute_sql_file")
    def test_create_bd_base_exception(self, mock_exec_sql, mock_session):
        """Test create_bd_base avec exception"""
        mock_db = MagicMock()
        mock_session.return_value = mock_db
        mock_exec_sql.side_effect = Exception("SQL error")
        
        create_bd_base()
        
        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()
